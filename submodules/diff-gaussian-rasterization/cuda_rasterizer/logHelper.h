#include <math.h>
#include <cstdio>

static float logFactor = 255.0f / logf(255.0f + 1.0f);

class LogHelper
{
public:
  static const char *JOB_LOG_PATH;

  __device__ __host__ static void appendLog(const char *message)
  {
#ifdef __CUDA_ARCH__
    // Device code: file I/O unavailable, fall back to printf
    printf("%s\n", message);
#else
    // Host code: append to log file
    FILE *f = fopen(JOB_LOG_PATH, "a");
    if (f)
    {
      fprintf(f, "%s\n", message);
      fclose(f);
    }
#endif
  }
  __device__ __host__ static float Log2sRGB(float value)
  {
    float linear = Log2Linear(value);
    float srgb = Linear2sRGB(linear);
    return srgb;
  }
  __device__ __host__ static float Log2Linear(float value)
  {
    // float val = 255.0f * (expf(value / 255.0f) - 1.0f) / (expf(1.0f) - 1.0f);
    float val = expf(value);
    return val;
  }
  __device__ __host__ static float sRGB2Linear(float value)
  {
    float val = value / 255.0f;
    if (val <= 0.04045f)
      return val / 12.92f;
    else
      return powf((val + 0.055f) / 1.055f, 2.4f);
  }
  __device__ __host__ static float Linear2sRGB(float value)
  {
    float res = powf(value, 1.0f / 2.2f);
    return res;
    // if (value <= 0.0031308f)
    //   return value * 12.92f * 255.0f;
    // else
    //   return (1.055f * powf(value, 1.0f / 2.4f) - 0.055f) * 255.0f;
  }
  __device__ __host__ static float Linear2Log(float value)
  {
    float val = value / 255.0f;
    return logf(val);
  }
};

inline const char *LogHelper::JOB_LOG_PATH = "~/jobLog.txt";
