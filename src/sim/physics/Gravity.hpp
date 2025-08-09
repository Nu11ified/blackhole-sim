#pragma once

#include <cmath>
#include "sim/math/Vec.hpp"

namespace sim::physics {

struct GravityParams {
    double gravitationalConstant;   // G (scaled for simulation)
    double blackHoleMass;           // M
    double softeningRadius;         // epsilon to avoid singularity at r=0
    double relativisticBendFactor;  // dimensionless accel tweak multiplier; 0 disables accel tweak
    double speedOfLight;            // c in simulation units
    double bendingAngleScale;       // multiplier applied to delta-theta (1.0 = physical scale)

    static GravityParams fromGM(double G, double M,
                                double epsilon = 1e-3,
                                double accelBend = 0.0,
                                double c = 1.0,
                                double angleBendScale = 1.0) {
        return GravityParams{G, M, epsilon, accelBend, c, angleBendScale};
    }
};

// Newtonian acceleration with optional softening and an adjustable multiplicative tweak for bending.
// a(p) = - G M / r^3 * p, with r softened by epsilon.
inline Vec2 computeAcceleration(const Vec2& position, const GravityParams& params) {
    const double epsilon = params.softeningRadius;
    const double r2 = position.squaredNorm() + epsilon * epsilon;
    const double r = std::sqrt(r2);
    const double inv_r3 = 1.0 / (r2 * r);

    Vec2 acceleration = - (params.gravitationalConstant * params.blackHoleMass) * inv_r3 * position;

    // Relativistic bending tweak (simple multiplier). Keeping it off by default (0 => multiplier 1).
    const double tweakMultiplier = 1.0 + params.relativisticBendFactor;
    return tweakMultiplier * acceleration;
}

// Relativistic bending per-frame small-angle approximation:
// Δθ ≈ (4 G M / (c^2 r)) * (Δt / r) = 4 G M Δt / (c^2 r^2)
inline double computeRelativisticBendingDeltaTheta(const Vec2& position,
                                                   const GravityParams& params,
                                                   double deltaTimeSeconds) {
    const double epsilon = params.softeningRadius;
    const double r2 = position.squaredNorm() + epsilon * epsilon;
    const double c2 = params.speedOfLight * params.speedOfLight;
    const double base = 4.0 * params.gravitationalConstant * params.blackHoleMass * deltaTimeSeconds;
    const double deltaTheta = (r2 > 0.0) ? (base / (c2 * r2)) : 0.0;
    return params.bendingAngleScale * deltaTheta;
}

// Rotate velocity by a signed angle using a 2D rotation.
inline Vec2 rotateVelocity(const Vec2& velocity, double signedAngleRadians) {
    const double c = std::cos(signedAngleRadians);
    const double s = std::sin(signedAngleRadians);
    return Vec2(c * velocity.x() - s * velocity.y(),
                s * velocity.x() + c * velocity.y());
}

// Photon sphere radius for Schwarzschild BH: r_photon = 3 G M / c^2
inline double photonSphereRadius(const GravityParams& params) {
    const double c2 = params.speedOfLight * params.speedOfLight;
    return 3.0 * params.gravitationalConstant * params.blackHoleMass / c2;
}

// Capture condition: r < r_photon
inline bool isCapturedByPhotonSphere(const Vec2& position, const GravityParams& params) {
    const double r = position.norm();
    return r < photonSphereRadius(params);
}

} // namespace sim::physics


