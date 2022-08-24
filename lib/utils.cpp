#include <cstring>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/csrc/api/include/torch/serialize.h>

#include "utils.h"



static std::vector<char> get_the_bytes (std::string filename);
static int               symbol_to_id  (char32_t c);



static std::vector<char>
get_the_bytes(std::string filename)
{
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> bytes(
                          (std::istreambuf_iterator<char>(input)),
                          (std::istreambuf_iterator<char>()));

  input.close();
  return bytes;
}



static int
symbol_to_id (char32_t c)
{
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



at::Tensor
de_norm(at::Tensor &x, at::Tensor &mean, at::Tensor &std)
{
  torch::Tensor de_norm_x =  mean + (std * x);

  return torch::where(x == 0.0, 0.0, de_norm_x);
}



at::Tensor
load_tensor(const std::string &filename)
{
  std::vector<char> f = get_the_bytes(filename);
  torch::Tensor tensor = torch::pickle_load(f).toTensor();

  return tensor;
}



std::vector<int>
conv_str (const std::u32string &str)
{
  std::vector<int> out (str.size());
  size_t j = 0;
  for (size_t i = 0; i < str.size(); i++)
    {
      char32_t c = str[i];
      if (c != ' ')
        out[j++] = symbol_to_id (c);
    }

  out.resize(j);
  return out;
}
