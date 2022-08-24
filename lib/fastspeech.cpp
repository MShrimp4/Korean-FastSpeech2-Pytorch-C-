#include <torch/script.h> // One-stop header.

#include <filesystem>

#include <iostream>
#include <memory>

#include <codecvt>
#include <string>
#include <libg2pk.h>

#include "utils.h"
#include "misc.h"
#include "fastspeech.h"

#define MAX_WAV_VALUE 32768.0
#define SAMPLING_RATE 22050
#define HOP_LENGTH 256



using namespace FS2;



FastSpeech::FastSpeech()
{
  priv = NULL;
}
FastSpeech::~FastSpeech()
{
}



class FastSpeech::pd
{
public:
  pd(const char *path);
  ~pd() {delete g2pk;}
  torch::jit::script::Module fastspeech, vocgan;
  at::Tensor mean_mel , std_mel;
  G2PK::G2K* g2pk;
};



bool
FastSpeech::load_data(const char *path)
{
  try
    {
      priv = std::unique_ptr<pd> (new pd(path));
    }
  catch (const c10::Error& e)
    {
      std::cerr << "error loading the model\n";
      return false;
    }
  catch (...)
    {
      std::cerr << "failed init\n";
      return false;
    }

  return true;
}
FastSpeech::pd::pd (const char *path)
{
  std::filesystem::path root = path;
  std::filesystem::path fs = root / "fastspeech.pt";
  std::filesystem::path vg = root / "vocgan.pt";
  std::filesystem::path mn = root / "mean.pt";
  std::filesystem::path st = root / "std.pt";

  if (!fs.has_filename()
   || !vg.has_filename()
   || !mn.has_filename()
   || !st.has_filename())
    throw std::runtime_error("One of the FastSpeech files does not exist\n");
  fastspeech = torch::jit::load (fs);
  vocgan     = torch::jit::load (vg);
  mean_mel   = load_tensor(mn).reshape({1,-1});
  std_mel    = load_tensor(st).reshape({1,-1});
  g2pk = new G2PK::G2K ();
}



std::vector<int16_t>
FastSpeech::synthesize(const char *text)
{

  if (!priv)
    {
      std::cerr << "FastSpeech data not loaded\n";
      return std::vector<int16_t>(0);
    }
  torch::jit::script::Module& fastspeech = priv->fastspeech;
  torch::jit::script::Module& vocgan     = priv->vocgan;

  at::Tensor& mean_mel = priv->mean_mel;
  at::Tensor& std_mel  = priv->std_mel;

  G2PK::G2K* g2pk = priv->g2pk;

  c10::InferenceMode guard;

  std::u32string decomposed = g2pk->decompose(
                                G2PK::u8_to_u32(
                                  g2pk->convert(text)));

  std::vector<int> input_vec = conv_str (decomposed);
  std::vector<int> len_vec   = {(int) input_vec.size()};
  at::Tensor input_tensor = torch::tensor(input_vec).unsqueeze(0);
  at::Tensor len_tensor   = torch::tensor(len_vec);

  std::vector<torch::jit::IValue> input_fs;
  input_fs.push_back(input_tensor);
  input_fs.push_back(len_tensor);

  at::Tensor output_fs = fastspeech.forward(input_fs).toTensor();

  std::vector<torch::jit::IValue> input_vg;
  output_fs = de_norm(output_fs, mean_mel, std_mel);
  input_vg.push_back(output_fs.transpose(1,2));

  at::Tensor output_vg = vocgan.forward(input_vg).toTensor();
  output_vg = output_vg.squeeze();
  int64_t len = at::size(output_vg, 0);
  output_vg = output_vg.narrow(0, 0, len - HOP_LENGTH * 10);
  output_vg *= MAX_WAV_VALUE;
  output_vg = output_vg.clamp(-MAX_WAV_VALUE, MAX_WAV_VALUE-1);
  output_vg = output_vg.toType(at::kShort);

  output_vg = output_vg.contiguous();
  std::vector<int16_t> v(output_vg.data_ptr<int16_t>(),
                         output_vg.data_ptr<int16_t>() + output_vg.numel());

  return v;
}



void
FastSpeech::save_wav  (std::vector<int16_t>  raw,
                       const char           *path)
{
  create_wav(raw, path);
}
