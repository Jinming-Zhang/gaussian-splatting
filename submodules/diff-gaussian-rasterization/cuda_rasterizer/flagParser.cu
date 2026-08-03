#include "flagParser.h"

// namespace
// {
__device__ int flag_original = 0;
__device__ int flag_render_sh_base_only = 1 << 0;
__device__ int flag_render_sh_higher_order = 1 << 1;
__device__ int flag_log_learning = 1 << 2;
// }

__global__ void parseRenderMode(int flag, int *out)
{
  if (flag_original | flag == 0)
  {
    *out = 0;
  }
  else if (flag_render_sh_base_only & flag)
  {
    *out = 1;
  }
  else if (flag_render_sh_higher_order & flag)
  {
    *out = 2;
  }
}

__global__ void parseLearningMode(int flag, int *out)
{
  if (flag_log_learning & flag)
  {
    *out = 1;
  }
  else
  {
    *out = 0;
  }
}
