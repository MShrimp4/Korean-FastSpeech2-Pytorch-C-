#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <chrono>

#include "../lib/fastspeech.h"



int main(int argc, const char* argv[]) {
  FS2::FastSpeech fs = FS2::FastSpeech();
  if (!fs.load_data("."))
    return 1;

  std::string input = "";
  while (true) {
    std::cout << u8"여기에 텍스트 입력:" << std::endl;

    std::getline (std::cin, input);

    std::cout << u8"원본 텍스트:" << input << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int16_t> raw = fs.synthesize(input.data());
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    std::cout << u8"합성 완료 - " << microseconds << "us" << std::endl;

    fs.save_wav(raw, "./test.wav");
  }
}
