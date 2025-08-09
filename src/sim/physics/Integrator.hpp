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

// Single-step integrator following Step 1.5
// 1) r, capture
// 2) Δθ and rotate v
// 3) p += v dt
inline void stepLightRay(LightRay& ray, const GravityParams& params, double deltaTimeSeconds) {
    if (ray.captured) return;

    const double r = ray.position.norm();
    if (r < photonSphereRadius(params)) {
        ray.captured = true;
        return;
    }

    // Rotate velocity by relativistic bending approximation
    const double dtheta = computeRelativisticBendingDeltaTheta(ray.position, params, deltaTimeSeconds);
    ray.velocity = rotateVelocity(ray.velocity, dtheta);

    // Maintain |v| = c for photons in our model
    enforceLightSpeed(ray.velocity, params.speedOfLight);

    // Update position
    ray.position += ray.velocity * deltaTimeSeconds;
}

} // namespace sim::physics


