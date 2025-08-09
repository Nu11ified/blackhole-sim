#pragma once

#include <Eigen/Dense>
#include <random>

namespace sim::ml {

struct RFFConfig {
    int numFeatures;    // D, must be even ideally (cos/sin pair), but we'll use only cos+bias here
    float gamma;        // kernel width parameter: K(x,y)=exp(-gamma||x-y||^2)
    unsigned int seed;
};

struct RandomFourierFeatures {
    // Map R^d -> R^D: phi(x) = sqrt(2/D) * cos(W x + b)
    Eigen::MatrixXf W; // D x d
    Eigen::VectorXf b; // D
    float scale;        // sqrt(2/D)
    bool fitted{false};

    void fit(int inputDim, const RFFConfig& cfg) {
        W.resize(cfg.numFeatures, inputDim);
        b.resize(cfg.numFeatures);
        scale = std::sqrt(2.0f / static_cast<float>(cfg.numFeatures));

        std::mt19937 rng(cfg.seed);
        std::normal_distribution<float> ndist(0.0f, std::sqrt(2.0f * cfg.gamma));
        std::uniform_real_distribution<float> udist(0.0f, 2.0f * static_cast<float>(M_PI));

        for (int i = 0; i < cfg.numFeatures; ++i) {
            for (int j = 0; j < inputDim; ++j) {
                W(i, j) = ndist(rng);
            }
            b[i] = udist(rng);
        }
        fitted = true;
    }

    Eigen::VectorXf transform(const Eigen::VectorXf& x) const {
        Eigen::VectorXf z = W * x; // D
        z.array() += b.array();
        z = z.array().cos();
        return scale * z;
    }
};

} // namespace sim::ml


