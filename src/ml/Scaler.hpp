#pragma once

#include <Eigen/Dense>
#include <vector>

namespace sim::ml {

struct FeatureScaler {
    Eigen::VectorXf mean;
    Eigen::VectorXf std;
    bool fitted{false};

    void fit(const std::vector<Eigen::VectorXf>& features) {
        if (features.empty()) { fitted = false; return; }
        const int dim = static_cast<int>(features.front().size());
        mean = Eigen::VectorXf::Zero(dim);
        std = Eigen::VectorXf::Zero(dim);
        for (const auto& x : features) {
            mean += x;
        }
        mean /= static_cast<float>(features.size());
        for (const auto& x : features) {
            Eigen::VectorXf diff = x - mean;
            std.array() += diff.array().square();
        }
        std = (std / static_cast<float>(features.size())).array().sqrt();
        // Avoid zeros
        for (int i = 0; i < dim; ++i) {
            if (std[i] < 1e-6f) std[i] = 1.0f;
        }
        fitted = true;
    }

    Eigen::VectorXf transform(const Eigen::VectorXf& x) const {
        if (!fitted) return x;
        return (x - mean).cwiseQuotient(std);
    }
};

} // namespace sim::ml


