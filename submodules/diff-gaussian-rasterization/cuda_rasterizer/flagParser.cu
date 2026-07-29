#include "flagParser.h"

namespace
{
  int flag_original = 0;
  int flag_render_sh_base_only = 1 << 0;
  int flag_render_sh_higher_order = 1 << 1;
  int flag_log_learning = 1 << 2;
}

__global__ int parseRenderMode(int flag)
{
  if (flag_original | flag == 0)
  {
    return 0;
  }
  if (flag_render_sh_base_only & flag)
  {
    return 1;
  }
  if (flag_render_sh_higher_order & flag)
  {
    return 2;
  }
}

__global__ int parseLearningMode(int flag)
{
  if (flag_log_learning & flag)
  {
    return 1;
  }
  return 0;
}
