#pragma once
#include <map>
#include <tuple>
#include <vector>

#include <torch/extension.h>
// means3D must point to CUDA device memory laid out as P contiguous (x, y, z) triples.
// Returns groups of gaussian indices sharing the same unit-cell in world space.
std::map<std::tuple<int, int, int>, std::vector<int>> GroupGaussians(const torch::Tensor &means3D);
std::vector<float> CalcPerGaussianNeighborsRefLoss(const torch::Tensor &means3D, const torch::Tensor &reflect_factors);
std::vector<float> CalcPerGaussianNeighborsRefLossCUDA(const torch::Tensor &means3D, const torch::Tensor &reflect_factors);

std::unordered_map<std::tuple<int, int, int>, std::vector<size_t>, struct TupleHash> GetGaussianGroups(const torch::Tensor &means3D);

struct TupleHash {
    template <typename T>
    static void hash_combine(std::size_t& seed, const T& v) {
        seed ^= std::hash<T>{}(v) + 0x9e3779b97f4a7c15ULL
              + (seed << 6) + (seed >> 2);
    }

    std::size_t operator()(const std::tuple<int, int, int>& t) const noexcept {
        std::size_t h = 0;
        hash_combine(h, std::get<0>(t));
        hash_combine(h, std::get<1>(t));
        hash_combine(h, std::get<2>(t));
        return h;
    }
};
