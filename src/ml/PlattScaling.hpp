#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace sim::ml {

// Fit logistic: P(y=1|s) = 1 / (1 + exp(A s + B)) on decision scores s with labels y in {+1,-1}
struct PlattScaler {
    float A{0.0f};
    float B{0.0f};
    bool fitted{false};

    static float sigmoid(float t) { return 1.0f / (1.0f + std::exp(t)); }

    void fit(const std::vector<float>& scores, const std::vector<int>& labels, int iters = 200, float lr = 0.01f) {
        if (scores.empty()) { fitted = false; return; }
        A = 0.0f; B = 0.0f;
        for (int k = 0; k < iters; ++k) {
            float gA = 0.0f, gB = 0.0f;
            for (size_t i = 0; i < scores.size(); ++i) {
                float y = labels[i] > 0 ? 1.0f : 0.0f; // logistic targets in {0,1}
                float p = sigmoid(A * scores[i] + B);
                float e = p - y;
                gA += e * scores[i];
                gB += e;
            }
            A -= lr * gA / static_cast<float>(scores.size());
            B -= lr * gB / static_cast<float>(scores.size());
        }
        fitted = true;
    }

    float probability(float score) const {
        if (!fitted) return 0.5f;
        return sigmoid(A * score + B);
    }
};

} // namespace sim::ml


