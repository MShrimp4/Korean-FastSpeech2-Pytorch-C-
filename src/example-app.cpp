#include <torch/script.h> // One-stop header.
#include <torch/csrc/api/include/torch/serialize.h>

#include <iostream>
#include <memory>

#define MAX_WAV_VALUE 32768.0
#define HOP_LENGTH 256
#define SAMPLING_RATE 22050

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

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> input_fs;
  input_fs.push_back(torch::randint(44,{1, 42}));
  input_fs.push_back(torch::tensor({42}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output_fs = fastspeech.forward(input_fs).toTensor();
  std::cout << output_fs << std::endl;

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

  std::cout << output_vg << std::endl;
}
