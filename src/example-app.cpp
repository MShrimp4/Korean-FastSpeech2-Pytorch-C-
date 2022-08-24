#include <torch/script.h> // One-stop header.
#include <torch/csrc/api/include/torch/serialize.h>

#include <fstream> //DELETE THIS
#include <iostream>
#include <memory>

#include <codecvt>
#include <cstring>
#include <string>
#include <libg2pk.h>

#define MAX_WAV_VALUE 32768.0
#define HOP_LENGTH 256
#define SAMPLING_RATE 22050

/*************************
 TESTONLY
**************************/
typedef struct WAV_HEADER {
  /* RIFF Chunk Descriptor */
  uint8_t RIFF[4] = {'R', 'I', 'F', 'F'}; // RIFF Header Magic header
  uint32_t ChunkSize;                     // RIFF Chunk Size
  uint8_t WAVE[4] = {'W', 'A', 'V', 'E'}; // WAVE Header
  /* "fmt" sub-chunk */
  uint8_t fmt[4] = {'f', 'm', 't', ' '}; // FMT header
  uint32_t Subchunk1Size = 16;           // Size of the fmt chunk
  uint16_t AudioFormat = 1; // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM
                            // Mu-Law, 258=IBM A-Law, 259=ADPCM
  uint16_t NumOfChan = 1;   // Number of channels 1=Mono 2=Sterio
  uint32_t SamplesPerSec = SAMPLING_RATE;   // Sampling Frequency in Hz
  uint32_t bytesPerSec = SAMPLING_RATE * sizeof(int16_t); // bytes per second
  uint16_t blockAlign = sizeof(int16_t);  // 2=16-bit mono, 4=16-bit stereo
  uint16_t bitsPerSample = 16;      // Number of bits per sample
  /* "data" sub-chunk */
  uint8_t Subchunk2ID[4] = {'d', 'a', 't', 'a'}; // "data"  string
  uint32_t Subchunk2Size;                        // Sampled data length
} wav_hdr;

int create_wav(std::vector<int16_t> &pcm, const std::string &fname) {
  static_assert(sizeof(wav_hdr) == 44, "");

  uint32_t fsize = pcm.size() * sizeof(int16_t);

  printf("file size: %u\n", fsize);

  wav_hdr wav;
  wav.ChunkSize = fsize + sizeof(wav_hdr) - 8;
  wav.Subchunk2Size = fsize + sizeof(wav_hdr) - 44;

  std::ofstream out(fname, std::ios::binary);
  out.write(reinterpret_cast<const char *>(&wav), sizeof(wav));

  for (int16_t d : pcm) {
    out.write(reinterpret_cast<char *>(&d), sizeof(int16_t));
  }

  return 0;
}
/*******************
 TESTONLY
 *******************/

/*def de_norm(x, mean, std):
    zero_idxs = torch.where(x == 0.0)[0]
    x = mean + std * x
    x[zero_idxs] = 0.0
    return x*/
at::Tensor de_norm(at::Tensor &x, at::Tensor &mean, at::Tensor &std) {
  torch::Tensor de_norm_x =  mean + (std * x);

  return torch::where(x == 0.0, 0.0, de_norm_x);
}

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

at::Tensor load_tensor(const std::string &filename) {
  std::vector<char> f = get_the_bytes(filename);
  torch::Tensor tensor = torch::pickle_load(f).toTensor();

  return tensor;
}

/*
def symbol_to_id (char):
    cint = ord(char)
    if      0x1100 <= cint < 0x1113:
        return cint - 0x1100 + 2
    elif 0x1161 <= cint < 0x1176:
        return cint - 0x1161 + (0x1113 - 0x1100) + 2
    elif 0x11A8 <= cint < 0x11C3:
        return cint - 0x11A8 + (0x1113 - 0x1100) + (0x1176 - 0x1161) + 2
    elif char in PUNC:
        return PUNC.find(char) + (0x1113 - 0x1100) + (0x1176 - 0x1161) + (0x11C3 - 0x11A8) + 2
    else:
        return 0*/

int symbol_to_id (char32_t c){
  if (0x1100 <= c && c < 0x1113)
    return c - 0x1100 + 2;
  else if (0x1161 <= c && c < 0x1176)
    return c - 0x1161 + (0x1113 - 0x1100) + 2;
  else if (0x11A8 <= c && c < 0x11C3)
    return c - 0x11A8 + (0x1113 - 0x1100) + (0x1176 - 0x1161) + 2;

  static const std::string punc = "!\'(),-.:;?";
  std::string::size_type n = punc.find(c);
  if (n != std::string::npos)
    return n + (0x1113 - 0x1100) + (0x1176 - 0x1161) + (0x11C3 - 0x11A8) + 2;
  else
    return 0;
}

std::vector<int> conv_str (const std::u32string &str) {
  std::vector<int> out (str.size());
  size_t j = 0;
  for (size_t i = 0; i < str.size(); i++){
    char32_t c = str[i];
    if (c != ' ')
      out[j++] = symbol_to_id (c);
  }

  out.resize(j);
  return out;
}

int main(int argc, const char* argv[]) {
  if (argc != 5) {
    std::cerr << "usage : " << argv[0]
              << " fastspeech.pt vocgan.pt mean.pt std.pt" << std::endl;
    return -1;
  }


  torch::jit::script::Module fastspeech, vocgan;
  at::Tensor mean_mel , std_mel;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    fastspeech = torch::jit::load(argv[1]);
    vocgan     = torch::jit::load(argv[2]);

    mean_mel = load_tensor(argv[3]).reshape({1,-1});
    std_mel  = load_tensor(argv[4]).reshape({1,-1});
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

  c10::InferenceMode guard;
  G2PK::G2K g2pk = G2PK::G2K ();

  std::string input = "";
  while (true) {
    std::cout << "여기에 텍스트 입력:" << std::endl;

    std::getline (std::cin, input);

    std::cout << "원본 텍스트:" << input << std::endl;

    std::u32string decomposed = g2pk.decompose(G2PK::u8_to_u32(g2pk.convert(input)));
    std::vector<int> input_vec = conv_str (decomposed);
    std::vector<int> len_vec = {(int) input_vec.size()};
    at::Tensor input_tensor = torch::tensor(input_vec).unsqueeze(0);
    at::Tensor len_tensor = torch::tensor(len_vec);

    std::cout << input_tensor << std::endl;

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> input_fs;
    input_fs.push_back(input_tensor);
    input_fs.push_back(len_tensor);

    // Execute the model and turn its output into a tensor.
    at::Tensor output_fs = fastspeech.forward(input_fs).toTensor();
    //std::cout << output_fs << std::endl;

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

    //std::cout << output_vg << std::endl;

    output_vg = output_vg.contiguous();
    std::vector<int16_t> v(output_vg.data_ptr<int16_t>(),
                           output_vg.data_ptr<int16_t>() + output_vg.numel());

    create_wav(v, "test.wav");
  }

}
