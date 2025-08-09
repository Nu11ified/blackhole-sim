#pragma once

#include <Eigen/Dense>

namespace sim {

// 2D vector aliases for positions, velocities, accelerations
using Vec2 = Eigen::Vector2d;

// 2x2 matrix alias (handy later for Jacobians/linearization)
using Mat2 = Eigen::Matrix2d;

} // namespace sim


