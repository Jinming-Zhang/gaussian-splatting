#include "VectorCompare.h"
#include <vector>
#include <tuple>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <string>

const float tolerance = 1e-5f;
ComparisonResult CompareVectors(const std::vector<float> &vec1, const std::vector<float> &vec2)
{
  assert(vec1.size() == vec2.size() && "Vectors must be of the same size for comparison.");

  std::vector<std::tuple<int, float, float>> differences;
  float totalDifference = 0.0f;
  for (size_t i = 0; i < vec1.size(); ++i)
  {
    float diff = std::abs(vec1[i] - vec2[i]);
    if (diff > tolerance)
    {
      differences.push_back({i, vec1[i], vec2[i]});
      totalDifference += diff;
    }
  }

  const char *home = std::getenv("HOME");
  if (home != nullptr)
  {
    std::string path = std::string(home) + "/difflog.txt";
    std::ofstream out(path, std::ios::app);
    if (out.is_open())
    {
      out << "CompareVectors: size=" << vec1.size()
          << " diffs=" << differences.size()
          << " totalDifference=" << totalDifference << "\n";
      for (const auto &d : differences)
      {
        out << "  index=" << std::get<0>(d)
            << " vec1=" << std::get<1>(d)
            << " vec2=" << std::get<2>(d)
            << " |diff|=" << std::abs(std::get<1>(d) - std::get<2>(d)) << "\n";
      }
    }
  }

  return {totalDifference, differences};
}
