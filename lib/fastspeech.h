#include <vector>
#include <memory>

namespace FS2
{
  class FastSpeech
  {
  public:
    FastSpeech();
    ~FastSpeech();

    const int sampling_rate = 22050;

    bool                 load_data (const char *path);
    std::vector<int16_t> synthesize(const char *text);
    void                 save_wav  (std::vector<int16_t>  raw,
                                    const char           *path);
  private:
    class pd;
    std::unique_ptr<pd> priv;
  };
}
