at::Tensor       de_norm     (at::Tensor &x,
                              at::Tensor &mean,
                              at::Tensor &std);
at::Tensor       load_tensor (const std::string &filename);
std::vector<int> conv_str    (const std::u32string &str);
