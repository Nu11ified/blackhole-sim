#pragma once

#include "sim/math/Vec.hpp"
#include "sim/physics/Gravity.hpp"

namespace sim::physics {

struct LightRay {
    Vec2 position;     // p
    Vec2 velocity;     // v (intended |v| ≈ c)
    bool captured{false};
};

inline void enforceLightSpeed(Vec2& velocity, double c) {
    const double speed = velocity.norm();
    if (speed > 0.0 && c > 0.0) {
        velocity *= (c / speed);
    }
}

// GR-inspired curvature-based integrator for null geodesics in a Schwarzschild field (2D slice).
// Keeps |v| = c and turns velocity toward the black hole with local curvature
// omega ≈ (R_s c) / r^2, which integrates to ~4GM/(c^2 b) far from the BH.
inline void stepLightRay(LightRay& ray, const GravityParams& params, double deltaTimeSeconds) {
    if (ray.captured) return;

    const double rs = schwarzschildRadius(params);
    const double r = ray.position.norm();

    // Event horizon capture at r <= R_s
    if (r <= rs) {
        ray.captured = true;
        return;
    }

    // Capture cone criterion using critical impact parameter b_c
    // b = |r x v| / |v| in 2D; if inward (vr<0) and b < b_c, photon cannot escape
    {
        const double vnorm = ray.velocity.norm();
        if (vnorm > 0.0 && r > 0.0) {
            const double vr = ray.velocity.dot(ray.position) / r;
            const double b = std::abs(ray.position.x() * ray.velocity.y() - ray.position.y() * ray.velocity.x()) / vnorm;
            const double bc = criticalImpactParameter(params);
            if (vr < 0.0 && b < bc) {
                ray.captured = true;
                return;
            }
        }
    }

    // Compute unit velocity and perpendicular steering direction toward -r_hat
    Vec2 vhat = ray.velocity.normalized();
    Vec2 rhat = (r > 0.0) ? (ray.position / r) : Vec2(1.0, 0.0);
    Vec2 inward = -rhat;
    // Perpendicular component of 'inward' relative to current motion
    Vec2 perp = inward - inward.dot(vhat) * vhat;
    const double perpNorm = perp.norm();
    if (perpNorm > 1e-18) perp /= perpNorm; else perp.setZero();

    // Local angular speed and small rotation of velocity direction
    const double omega = localCurvatureOmega(ray.position, params);
    const double dtheta = omega * deltaTimeSeconds;
    // Update vhat in the plane: vhat' = vhat cos dθ + perp sin dθ
    const double cth = std::cos(dtheta);
    const double sth = std::sin(dtheta);
    Vec2 vhat_new = cth * vhat + sth * perp;
    // Guard numerical drift
    if (vhat_new.norm() > 0.0) vhat_new.normalize();

    // Maintain |v| = c for photons in our model
    ray.velocity = vhat_new * params.speedOfLight;

    // Update position
    ray.position += ray.velocity * deltaTimeSeconds;
}

} // namespace sim::physics
