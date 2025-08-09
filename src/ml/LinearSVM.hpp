#pragma once

#include <vector>
#include <Eigen/Dense>

#include "ml/DataCollection.hpp"

namespace sim::ml {

struct SVMHyperParams {
    float learningRate;   // eta
    float C_pos;          // class-weighted C for +1 class
    float C_neg;          // class-weighted C for -1 class
    int epochs;           // number of passes over the dataset
};

class LinearSVM {
public:
    LinearSVM() : bias_(0.0f) {}

    void train(const std::vector<DataPoint>& dataset, const SVMHyperParams& hp) {
        if (dataset.empty()) return;
        const int dim = static_cast<int>(dataset.front().features.size());
        if (weights_.size() != dim) weights_ = Eigen::VectorXf::Zero(dim);
        for (int epochIndex = 0; epochIndex < hp.epochs; ++epochIndex) {
            for (const DataPoint& dp : dataset) {
                const float yi = static_cast<float>(dp.label); // +1 or -1
                const float score = weights_.dot(dp.features) + bias_;
                const float margin = yi * score;
                if (margin >= 1.0f) {
                    // w <- w - eta * w
                    weights_ -= hp.learningRate * weights_;
                } else {
                    // class-weighted C
                    const float Cw = (yi > 0.0f) ? hp.C_pos : hp.C_neg;
                    // w <- w - eta * (w - Cw y x)
                    // b <- b + eta * Cw y
                    weights_ -= hp.learningRate * (weights_ - Cw * yi * dp.features);
                    bias_ += hp.learningRate * Cw * yi;
                }
            }
        }
    }

    float decisionFunction(const Eigen::VectorXf& x) const {
        if (weights_.size() == 0 || weights_.size() != x.size()) {
            return bias_;
        }
        return weights_.dot(x) + bias_;
    }

    int predictLabel(const Eigen::VectorXf& x) const {
        const float s = decisionFunction(x);
        return s >= 0.0f ? +1 : -1;
    }

    float accuracy(const std::vector<DataPoint>& dataset) const {
        if (dataset.empty()) return 0.0f;
        int correct = 0;
        for (const DataPoint& dp : dataset) {
            if (predictLabel(dp.features) == dp.label) ++correct;
        }
        return static_cast<float>(correct) / static_cast<float>(dataset.size());
    }

    const Eigen::VectorXf& weights() const { return weights_; }
    float bias() const { return bias_; }

private:
    Eigen::VectorXf weights_;
    float bias_;
};

} // namespace sim::ml


