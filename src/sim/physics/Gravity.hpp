#pragma once

#include <cmath>
#include "sim/math/Vec.hpp"

namespace sim::physics {

struct GravityParams {
    double gravitationalConstant;   // G (scaled for simulation)
    double blackHoleMass;           // M
    double softeningRadius;         // epsilon to avoid singularity at r=0
    double relativisticBendFactor;  // legacy tweak (kept for compat); not used in GR step
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

    // Legacy tweak multiplier retained for backward compatibility if nonzero
    const double tweakMultiplier = 1.0 + params.relativisticBendFactor;
    return tweakMultiplier * acceleration;
}

// Schwarzschild length scales
inline double schwarzschildRadius(const GravityParams& params) {
    // R_s = 2 G M / c^2
    const double c2 = params.speedOfLight * params.speedOfLight;
    return 2.0 * params.gravitationalConstant * params.blackHoleMass / c2;
}

// Photon sphere radius for Schwarzschild BH: r_photon = 3 G M / c^2 = 1.5 R_s
inline double photonSphereRadius(const GravityParams& params) {
    return 1.5 * schwarzschildRadius(params);
}

// GR-inspired local curvature rate for light: w = (R_s * c) / r^2
// Returns angular speed (rad/s) controlling how quickly the velocity direction turns toward -r_hat.
inline double localCurvatureOmega(const Vec2& position, const GravityParams& params) {
    const double rs = schwarzschildRadius(params);
    const double r2 = position.squaredNorm() + params.softeningRadius * params.softeningRadius;
    // Factor 2 to better match far-field GR deflection 2 R_s / b
    return params.bendingAngleScale * (2.0 * rs * params.speedOfLight) / std::max(r2, 1e-18);
}

// Critical impact parameter for capture from infinity: b_c = (3*sqrt(3)/2) * R_s
inline double criticalImpactParameter(const GravityParams& params) {
    return 0.5 * 3.0 * std::sqrt(3.0) * schwarzschildRadius(params);
}

// Rotate velocity by a signed angle using a 2D rotation.
inline Vec2 rotateVelocity(const Vec2& velocity, double signedAngleRadians) {
    const double c = std::cos(signedAngleRadians);
    const double s = std::sin(signedAngleRadians);
    return Vec2(c * velocity.x() - s * velocity.y(),
                s * velocity.x() + c * velocity.y());
}

// Capture condition: r < r_photon
inline bool isCapturedByPhotonSphere(const Vec2& position, const GravityParams& params) {
    const double r = position.norm();
    return r < photonSphereRadius(params);
}

} // namespace sim::physics
