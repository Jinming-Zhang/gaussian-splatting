#pragma once
#include <vector>
#include <tuple>


struct ComparisonResult
{
  float totalDifference;
  std::vector<std::tuple<int, float, float>> differences;
};

ComparisonResult CompareVectors(const std::vector<float> &vec1, const std::vector<float> &vec2);
