#pragma once

#include <vector>
#include <random>
#include <Eigen/Dense>

#include "sim/math/Vec.hpp"
#include "sim/physics/Gravity.hpp"
#include "sim/physics/Integrator.hpp"

namespace sim::ml {

struct DataPoint {
    Eigen::VectorXf features; // can be 4 (raw) or expanded (e.g., RFF)
    int label;                // +1 escape, -1 captured
};

inline Eigen::VectorXf featuresFromState(const sim::Vec2& position, const sim::Vec2& velocity) {
    Eigen::VectorXf x(4);
    x << static_cast<float>(position.x()), static_cast<float>(position.y()),
         static_cast<float>(velocity.x()), static_cast<float>(velocity.y());
    return x;
}

inline bool hasEscaped(const sim::Vec2& position, double maxRadius) {
    return position.norm() > maxRadius;
}

// Returns +1 if escaped, -1 if captured (or not captured within max steps and within bounds => treat as escaped)
inline int simulateOutcomeForInitialState(const sim::Vec2& initialPosition,
                                          const sim::Vec2& initialVelocity,
                                          const sim::physics::GravityParams& params,
                                          double deltaTimeSeconds,
                                          double maxRadius,
                                          int maxSteps) {
    sim::physics::LightRay ray{initialPosition, initialVelocity, false};
    for (int stepIndex = 0; stepIndex < maxSteps; ++stepIndex) {
        if (hasEscaped(ray.position, maxRadius)) {
            return +1;
        }
        if (ray.captured || sim::physics::isCapturedByPhotonSphere(ray.position, params)) {
            return -1;
        }
        sim::physics::stepLightRay(ray, params, deltaTimeSeconds);
    }
    // If we reach here, we didn't capture; consider it escaped for labeling purposes
    return +1;
}

struct RandomSamplerConfig {
    double positionRange; // sample x0,y0 ~ U(-positionRange, positionRange)
    double minRadius;     // if > 0, ensure sqrt(x0^2+y0^2) >= minRadius
    int numSamples;
    unsigned int seed;
};

inline std::vector<DataPoint> collectDatasetRandom(const sim::physics::GravityParams& params,
                                                   double deltaTimeSeconds,
                                                   double maxRadius,
                                                   int maxSteps,
                                                   const RandomSamplerConfig& sampler) {
    std::vector<DataPoint> dataset;
    dataset.reserve(static_cast<size_t>(sampler.numSamples));

    std::mt19937 rng(sampler.seed);
    std::uniform_real_distribution<double> posDist(-sampler.positionRange, sampler.positionRange);
    std::uniform_real_distribution<double> angleDist(0.0, 2.0 * M_PI);

    for (int i = 0; i < sampler.numSamples; ++i) {
        // Sample position in a square, resample if below minRadius
        sim::Vec2 p;
        do {
            p = sim::Vec2(posDist(rng), posDist(rng));
        } while (sampler.minRadius > 0.0 && p.norm() < sampler.minRadius);

        // Sample velocity as direction with magnitude c (photon)
        const double theta = angleDist(rng);
        sim::Vec2 v(params.speedOfLight * std::cos(theta), params.speedOfLight * std::sin(theta));

        const int label = simulateOutcomeForInitialState(p, v, params, deltaTimeSeconds, maxRadius, maxSteps);
        Eigen::VectorXf feat(4);
        feat << static_cast<float>(p.x()), static_cast<float>(p.y()),
                static_cast<float>(v.x()), static_cast<float>(v.y());
        DataPoint dp{feat, label};
        dataset.push_back(dp);
    }
    return dataset;
}

} // namespace sim::ml


