#include <assert.h>
#include <fstream>
#include <string>
#include <vector>

#include "misc.h"



#define SAMPLING_RATE 22050



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



int create_wav(std::vector<int16_t> &pcm, const std::string &fname)
{
  static_assert(sizeof(wav_hdr) == 44, "");

  uint32_t fsize = pcm.size() * sizeof(int16_t);

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
