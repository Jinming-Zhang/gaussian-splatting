#pragma once
#include <map>
#include <tuple>
#include <vector>

#include <torch/extension.h>
// means3D must point to CUDA device memory laid out as P contiguous (x, y, z) triples.
// Returns groups of gaussian indices sharing the same unit-cell in world space.
std::map<std::tuple<int, int, int>, std::vector<int>> GroupGaussians(const torch::Tensor &means3D);
