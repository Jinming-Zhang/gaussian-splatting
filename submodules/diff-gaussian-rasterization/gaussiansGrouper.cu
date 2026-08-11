#include "gaussiansGrouper.h"

#include <set>
#include <map>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <cmath>
#include <iostream>

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/device_vector.h>

namespace cg = cooperative_groups;

__global__ void calcNeighborLossSumCUDA(const float *means, const int P, const float *rfs, const int *indices, const size_t numIndicies, float *out)
{
  auto block = cg::this_thread_block();
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numIndicies)
  {
    return;
  }

  int clrChannel = 3;
  int gaussianIdx = indices[idx];
  int r_index = gaussianIdx * clrChannel;
  float r_1 = rfs[r_index];
  float r_2 = rfs[r_index + 1];
  float r_3 = rfs[r_index + 2];

  float refLossForGi = 0.0f;

  for (int j = 0; j < numIndicies; ++j)
  {
    int r_j_index = indices[j] * clrChannel;
    float r_j_1 = rfs[r_j_index];
    float r_j_2 = rfs[r_j_index + 1];
    float r_j_3 = rfs[r_j_index + 2];
    refLossForGi += fabsf(r_j_1 - r_1) + fabsf(r_j_2 - r_2) + fabsf(r_j_3 - r_3);
  }
  out[gaussianIdx] = refLossForGi;
}

std::map<std::tuple<int, int, int>, std::vector<int>> GroupGaussians(const torch::Tensor &means3D)
{
  std::map<std::tuple<int, int, int>, std::vector<int>> groups{};
  size_t elmtsSize = means3D.size(0);

  for (size_t i = 0; i < elmtsSize; ++i)
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

std::vector<float> CalcPerGaussianNeighborsRefLoss(const torch::Tensor &means3D, const torch::Tensor &reflect_factors)
{
  // std::unordered_map<std::tuple<int, int, int>, std::unordered_map<int, float>, TupleHash> groups{};

  // size_t elmtsSize = means3D.size(0);
  // for (size_t i = 0; i < elmtsSize; ++i)
  // {
  //   printf("Processing gaussian %d/%d\r", static_cast<int>(i) + 1, static_cast<int>(elmtsSize));
  //   int x = static_cast<int>(std::floor(means3D[i][0].item<float>()));
  //   int y = static_cast<int>(std::floor(means3D[i][1].item<float>()));
  //   int z = static_cast<int>(std::floor(means3D[i][2].item<float>()));
  //   std::tuple<int, int, int> gridCoord = std::make_tuple(x, y, z);
  //   if (groups.find(gridCoord) == groups.end())
  //   {
  //     groups[gridCoord] = std::unordered_map<int, float>{{i, 0.0}};
  //   }
  //   else
  //   {
  //     std::unordered_map<int, float> &gaussianIdx2RefLossMap = groups[gridCoord];
  //     float refLossForGi = 0.0;
  //     float r_i = reflect_factors[i].item<float>();
  //     for (const auto &kv : gaussianIdx2RefLossMap)
  //     {
  //       int j = kv.first;
  //       float r_j = reflect_factors[j].item<float>();

  //       float err = std::abs(r_j - r_i);
  //       refLossForGi += err;

  //       // gaussianIdx2RefLossMap[kv.first] += (err + 0.1);
  //       gaussianIdx2RefLossMap[kv.first] += (err);
  //     }
  //     gaussianIdx2RefLossMap.insert({i, refLossForGi});
  //   }
  // }

  // size_t elmtsSize = means3D.size(0);
  // const std::unordered_map<std::tuple<int, int, int>, std::vector<size_t>, struct TupleHash>& groups = GetGaussianGroups(means3D);

  // std::cout << "Finished grouping, number of groups: " << groups.size() << std::endl;

  // std::vector<float> perGaussianRefLoss(elmtsSize, 0.0);
  // for (const auto &kv : groups)
  // {
  //   const std::unordered_map<int, float> &gaussianIdx2RefLossMap = kv.second;
  //   for (const auto &kv2 : gaussianIdx2RefLossMap)
  //   {
  //     perGaussianRefLoss[kv2.first] = kv2.second;
  //   }
  // }

  // return perGaussianRefLoss;

  size_t elmtsSize = means3D.size(0);
  const std::unordered_map<std::tuple<int, int, int>, std::vector<size_t>, struct TupleHash> &groups = GetGaussianGroups(means3D);

  std::cout << "Finished grouping, number of groups: " << groups.size() << std::endl;

  std::vector<float> perGaussianRefLoss(elmtsSize, 0.0);
  int ind = 1;
  for (const auto &kv : groups)
  {
    const std::vector<size_t> &indices = kv.second;
    for (size_t i = 0; i < indices.size(); ++i)
    {
      std::cout << "calculate total r loss for " << ind << "\r";
      ++ind;
      size_t r_i_ind = indices[i];
      float r_i = reflect_factors[r_i_ind].item<float>();
      float refLossForGi = 0.0;
      for (size_t j = 0; j < indices.size(); ++j)
      {
        size_t r_j_ind = indices[j];
        float r_j = reflect_factors[r_j_ind].item<float>();
        refLossForGi += std::abs(r_j - r_i);
      }
      perGaussianRefLoss[r_i_ind] = refLossForGi;
    }
  }
  std::cout << std::endl;
  return perGaussianRefLoss;
}

std::vector<float> CalcPerGaussianNeighborsRefLossCUDA(const torch::Tensor &means3D, const torch::Tensor &reflect_factors)
{
  // std::unordered_map<std::tuple<int, int, int>, std::vector<size_t>, TupleHash> group2GaussianIndices{};

  // size_t elmtsSize = means3D.size(0);
  // for (size_t i = 0; i < elmtsSize; ++i)
  // {
  //   printf("Processing gaussian %d/%d\r", static_cast<int>(i) + 1, static_cast<int>(elmtsSize));
  //   int x = static_cast<int>(std::floor(means3D[i][0].item<float>()));
  //   int y = static_cast<int>(std::floor(means3D[i][1].item<float>()));
  //   int z = static_cast<int>(std::floor(means3D[i][2].item<float>()));
  //   std::tuple<int, int, int> gridCoord = std::make_tuple(x, y, z);
  //   if (group2GaussianIndices.find(gridCoord) == group2GaussianIndices.end())
  //   {
  //     group2GaussianIndices[gridCoord] = std::vector<size_t>{i};
  //   }
  //   else
  //   {
  //     group2GaussianIndices[gridCoord].push_back(i);
  //   }
  // }

  size_t elmtsSize = means3D.size(0);
  const std::unordered_map<std::tuple<int, int, int>, std::vector<size_t>, struct TupleHash> &group2GaussianIndices = GetGaussianGroups(means3D);

  // std::cout << "Finished grouping, number of groups: " << group2GaussianIndices.size() << std::endl;
  // std::cout << reflect_factors[0].item<float>() << ", " << reflect_factors[1].item<float>() << ", " << reflect_factors[2].item<float>() << std::endl;

  thrust::device_vector<float> perGaussianRefLoss(elmtsSize, 0.0);
  for (const auto &kv : group2GaussianIndices)
  {
    const std::vector<size_t> &gaussianIndices = kv.second;
    thrust::device_vector<int> indicesCUDA(gaussianIndices.begin(), gaussianIndices.end());
    int gridSize = (gaussianIndices.size() + 255) / 256;
    calcNeighborLossSumCUDA<<<gridSize, 256>>>(means3D.contiguous().data<float>(),
                                               means3D.size(0),
                                               reflect_factors.contiguous().data<float>(),
                                               thrust::raw_pointer_cast(indicesCUDA.data()),
                                               indicesCUDA.size(),
                                               thrust::raw_pointer_cast(perGaussianRefLoss.data()));
  }
  cudaError_t res = cudaDeviceSynchronize();
  if (res != cudaSuccess)
  {
    std::cout << "CUDA error: " << cudaGetErrorString(res) << std::endl;
  }
  // std::cout << perGaussianRefLoss[0] << ", " << perGaussianRefLoss[1] << ", " << perGaussianRefLoss[2] << std::endl;
  return std::vector<float>(perGaussianRefLoss.begin(), perGaussianRefLoss.end());
}

std::unordered_map<std::tuple<int, int, int>, std::vector<size_t>, struct TupleHash> GetGaussianGroups(const torch::Tensor &means3D)
{
  std::unordered_map<std::tuple<int, int, int>, std::vector<size_t>, TupleHash> group2GaussianIndices{};

  size_t elmtsSize = means3D.size(0);
  for (size_t i = 0; i < elmtsSize; ++i)
  {
    // printf("Processing gaussian %d/%d\r", static_cast<int>(i) + 1, static_cast<int>(elmtsSize));
    int x = static_cast<int>(std::floor(means3D[i][0].item<float>()));
    int y = static_cast<int>(std::floor(means3D[i][1].item<float>()));
    int z = static_cast<int>(std::floor(means3D[i][2].item<float>()));
    std::tuple<int, int, int> gridCoord = std::make_tuple(x, y, z);
    if (group2GaussianIndices.find(gridCoord) == group2GaussianIndices.end())
    {
      group2GaussianIndices[gridCoord] = std::vector<size_t>{i};
    }
    else
    {
      group2GaussianIndices[gridCoord].push_back(i);
    }
  }
  return group2GaussianIndices;
}
