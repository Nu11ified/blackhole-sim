#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Dense>

#include "ml/DataCollection.hpp"

namespace sim::ml {

struct TrainValSplit {
    std::vector<DataPoint> train;
    std::vector<DataPoint> val;
};

inline TrainValSplit splitTrainVal(const std::vector<DataPoint>& data, float valRatio = 0.2f, unsigned int seed = 123u) {
    std::vector<DataPoint> shuffled = data;
    std::mt19937 rng(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    size_t valCount = static_cast<size_t>(valRatio * shuffled.size());
    TrainValSplit out;
    out.val.insert(out.val.end(), shuffled.begin(), shuffled.begin() + valCount);
    out.train.insert(out.train.end(), shuffled.begin() + valCount, shuffled.end());
    return out;
}

inline float accuracyOn(const std::vector<DataPoint>& data, const std::function<int(const Eigen::VectorXf&)>& predict) {
    if (data.empty()) return 0.0f;
    int correct = 0;
    for (const auto& dp : data) {
        if (predict(dp.features) == dp.label) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(data.size());
}

} // namespace sim::ml


