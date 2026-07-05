#include "gaussiansGrouper.h"

#include <thrust/device_vector.h>
#include <set>
#include <map>
#include <tuple>
#include <vector>
#include <cmath>
#include <iostream>
#include <torch/extension.h>

std::map<std::tuple<int, int, int>, std::vector<int>> GroupGaussians(const torch::Tensor &means3D)
{
  std::map<std::tuple<int, int, int>, std::vector<int>> groups{};
  size_t elmtsSize = means3D.size(0);
  for(size_t i = 0; i < elmtsSize; ++i)
  {
    int x = static_cast<int>(std::floor(means3D[i][0].item<float>()));
    int y = static_cast<int>(std::floor(means3D[i][1].item<float>()));
    int z = static_cast<int>(std::floor(means3D[i][2].item<float>()));
    std::tuple<int, int, int> gridCoord = std::make_tuple(x, y, z);
    if (groups.find(gridCoord) == groups.end())
    {
      groups[gridCoord] = std::vector<int>{};
    }
    groups[gridCoord].push_back(static_cast<int>(i));
  }

  return groups;
}
