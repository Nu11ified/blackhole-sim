#include <raylib.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "sim/math/Vec.hpp"
#include "sim/physics/Gravity.hpp"
#include "sim/physics/Integrator.hpp"
#include "ml/DataCollection.hpp"
#include "ml/LinearSVM.hpp"
#include "ml/Scaler.hpp"
#include "ml/RFF.hpp"
#include "ml/PlattScaling.hpp"
#include "ml/TrainUtils.hpp"

static Vector2 worldToScreen(const sim::Vec2& world, int screenW, int screenH, float pixelsPerUnit) {
    const float sx = static_cast<float>(screenW) * 0.5f + static_cast<float>(world.x()) * pixelsPerUnit;
    const float sy = static_cast<float>(screenH) * 0.5f - static_cast<float>(world.y()) * pixelsPerUnit;
    return {sx, sy};
}

static sim::Vec2 screenToWorld(const Vector2& screen, int screenW, int screenH, float pixelsPerUnit) {
    const double x = (static_cast<double>(screen.x) - static_cast<double>(screenW) * 0.5) / static_cast<double>(pixelsPerUnit);
    const double y = (static_cast<double>(screenH) * 0.5 - static_cast<double>(screen.y)) / static_cast<double>(pixelsPerUnit);
    return sim::Vec2(x, y);
}

int main() {
    const int screenW = 1000;
    const int screenH = 800;
    InitWindow(screenW, screenH, "Blackhole Sim - Rendering Phase");
    SetTargetFPS(60);

    // Basic usage of our Eigen-based types
    sim::physics::LightRay ray{
        /*position*/ sim::Vec2(1.0, 0.0),
        /*velocity*/ sim::Vec2(0.0, 0.9),
        /*captured*/ false
    };
    sim::Vec2 acceleration(0.0, 0.0);

    sim::physics::GravityParams grav = sim::physics::GravityParams::fromGM(
        1.0,  // G
        1.0,  // M
        1e-3, // epsilon
        0.0,  // accel tweak factor (off)
        1.0,  // c
        2.5   // bending angle scale (tunable with [ and ])
    );

    // Rendering params
    float pixelsPerUnit = 30.0f; // wider view to see bending and plunge
    const float r_s = static_cast<float>(sim::physics::schwarzschildRadius(grav));
    const float r_ph = static_cast<float>(sim::physics::photonSphereRadius(grav));

    // Accretion disk state
    float diskAngle = 0.0f;              // radians
    const float diskAngularSpeed = 0.6f; // rad/s
    const float diskInner = 1.2f * r_s;
    const float diskOuter = 3.5f * r_s; // extend a bit for nicer visuals

    // Background stars in world space
    std::vector<sim::Vec2> starsWorld;
    starsWorld.reserve(600);
    std::mt19937 rng(12345u);
    const float halfWWorld = static_cast<float>(screenW) / (2.0f * pixelsPerUnit);
    const float halfHWorld = static_cast<float>(screenH) / (2.0f * pixelsPerUnit);
    std::uniform_real_distribution<float> xdist(-halfWWorld, halfWWorld);
    std::uniform_real_distribution<float> ydist(-halfHWorld, halfHWorld);
    for (int i = 0; i < 600; ++i) {
        starsWorld.emplace_back(static_cast<double>(xdist(rng)), static_cast<double>(ydist(rng)));
    }

    // Ray container (multiple rays)
    struct RenderRay { sim::physics::LightRay ray; Color color; std::vector<sim::Vec2> trail; bool wasCaptured{false}; };
    std::vector<RenderRay> rays;
    rays.reserve(256);

    // Dataset collected from spawned rays; train SVM periodically to validate physics
    sim::ml::LinearSVM svm;
    sim::ml::SVMHyperParams trainHp{.learningRate = 0.001f, .C_pos = 1.2f, .C_neg = 1.0f, .epochs = 10};
    sim::ml::FeatureScaler scaler;
    sim::ml::RandomFourierFeatures rff;
    sim::ml::RFFConfig rffCfg{.numFeatures = 512, .gamma = 0.5f, .seed = 7u};
    bool rffReady = false;
    sim::ml::PlattScaler platt;
    std::vector<sim::ml::DataPoint> spawnedDataset;
    float svmAcc = 0.0f;

    // Automatic emitter: spawn rays from a ring with mostly tangential directions
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * PI);
    std::uniform_real_distribution<float> jitterDist(-0.2f, 0.2f); // radians
    const float worldRadiusMax = std::min(halfWWorld, halfHWorld);
    float spawnRadius = worldRadiusMax * 0.95f; // near edge of screen
    const float spawnPerSecond = 18.0f; // populate scene a bit more
    double spawnAccumulator = 0.0;

    while (!WindowShouldClose()) {
        const double dt = 1.0 / 60.0;

        // Auto-spawn rays at a fixed rate
        spawnAccumulator += dt;
        const double spawnInterval = 1.0 / static_cast<double>(spawnPerSecond);
        while (spawnAccumulator >= spawnInterval) {
            spawnAccumulator -= spawnInterval;

            float ang = angleDist(rng);
            sim::Vec2 p0(spawnRadius * std::cos(ang), spawnRadius * std::sin(ang));
            // Tangential direction with clearer inward bias to curve around BH
            sim::Vec2 tangent(-std::sin(ang), std::cos(ang));
            sim::Vec2 inward = -p0.normalized();
            sim::Vec2 dir = (0.45 * tangent + 0.55 * inward); // slight inward bias to encourage capture
            dir.normalize();
            // Small jitter
            dir = sim::physics::rotateVelocity(dir, static_cast<double>(jitterDist(rng)));
            sim::Vec2 v0 = dir * grav.speedOfLight;

            // Predict and color (if SVM has seen some data)
            int pred = +1;
            if (!spawnedDataset.empty()) {
                Eigen::VectorXf x4 = sim::ml::featuresFromState(p0, v0);
                Eigen::VectorXf xs = scaler.fitted ? scaler.transform(x4) : x4;
                Eigen::VectorXf z = rffReady ? rff.transform(xs) : xs;
                pred = svm.predictLabel(z);
            }
            Color col = pred >= 0 ? (Color){80, 255, 120, 255} : (Color){255, 80, 120, 255};

            RenderRay rr;
            rr.ray = sim::physics::LightRay{p0, v0, false};
            rr.color = col;
            rr.trail.reserve(2000);
            rr.trail.push_back(p0);
            rays.push_back(std::move(rr));
            if (rays.size() > 256) {
                rays.erase(rays.begin(), rays.begin() + (rays.size() - 256));
            }

            // Label via simulation to build dataset and (periodically) retrain SVM
            int label = sim::ml::simulateOutcomeForInitialState(p0, v0, grav, 0.01, worldRadiusMax * 1.2, 4000);
            spawnedDataset.push_back(sim::ml::DataPoint{sim::ml::featuresFromState(p0, v0), label});
            if (spawnedDataset.size() % 100 == 0) {
                // Prepare scaler & RFF on raw features
                std::vector<Eigen::VectorXf> raw; raw.reserve(spawnedDataset.size());
                for (auto& dp : spawnedDataset) raw.push_back(dp.features);
                scaler.fit(raw);
                if (!rffReady) { rff.fit(4, rffCfg); rffReady = true; }

                // Build transformed dataset
                std::vector<sim::ml::DataPoint> transformed; transformed.reserve(spawnedDataset.size());
                for (const auto& dp : spawnedDataset) {
                    Eigen::VectorXf xs = scaler.transform(dp.features);
                    Eigen::VectorXf z = rff.transform(xs);
                    transformed.push_back(sim::ml::DataPoint{z, dp.label});
                }

                // Train/val split and training
                auto split = sim::ml::splitTrainVal(transformed, 0.2f, 123u);
                svm.train(split.train, trainHp);
                svmAcc = svm.accuracy(split.val);

                // Fit Platt scaling on validation scores
                std::vector<float> scores; scores.reserve(split.val.size());
                std::vector<int> labels; labels.reserve(split.val.size());
                for (const auto& dp : split.val) {
                    // Safety: ensure dims match (they should)
                    scores.push_back(svm.decisionFunction(dp.features));
                    labels.push_back(dp.label);
                }
                platt.fit(scores, labels, 200, 0.01f);
            }
        }

        // Spawn rays with mouse drag: start=down, dir=to current mouse
        static bool isDragging = false;
        static sim::Vec2 dragStartWorld(0.0, 0.0);
        static int previewPred = 0;
        Vector2 mouse = GetMousePosition();
        sim::Vec2 mouseWorld = screenToWorld(mouse, screenW, screenH, pixelsPerUnit);
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) { isDragging = true; dragStartWorld = mouseWorld; previewPred = 0; }
        if (isDragging) {
            sim::Vec2 dir = mouseWorld - dragStartWorld; if (dir.norm() > 1e-6) dir.normalize();
            sim::Vec2 vPreview = dir * grav.speedOfLight;
            // Use current scaler/RFF+SVM if available
            if (!spawnedDataset.empty()) {
                Eigen::VectorXf x4 = sim::ml::featuresFromState(dragStartWorld, vPreview);
                Eigen::VectorXf xs = scaler.fitted ? scaler.transform(x4) : x4;
                Eigen::VectorXf z = rffReady ? rff.transform(xs) : xs;
                previewPred = svm.predictLabel(z);
            } else {
                previewPred = +1;
            }
            if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
                RenderRay rr; rr.ray = sim::physics::LightRay{dragStartWorld, vPreview, false};
                rr.color = (previewPred >= 0) ? GREEN : RED; rr.trail.reserve(2000);
                rr.trail.push_back(rr.ray.position); // seed trail with starting point
                rays.push_back(std::move(rr)); if (rays.size() > 64) rays.erase(rays.begin(), rays.begin() + (rays.size() - 64));
                isDragging = false; previewPred = 0;
            }
        }

        // Integrate all rays
        for (auto& rr : rays) {
            if (!rr.ray.captured) {
                sim::physics::stepLightRay(rr.ray, grav, dt);
                rr.trail.push_back(rr.ray.position);
                if (rr.trail.size() > 2000) rr.trail.erase(rr.trail.begin(), rr.trail.begin() + 500);
                // Stop integrating if ray goes off visible world
                if (rr.ray.position.norm() > (worldRadiusMax * 1.1)) {
                    rr.ray.captured = true;
                }
            }
            // If newly captured, extend the trail to the horizon for visual continuity
            if (rr.ray.captured && !rr.wasCaptured) {
                rr.wasCaptured = true;
                rr.color = RED;
                sim::Vec2 p = rr.trail.empty() ? rr.ray.position : rr.trail.back();
                for (int i = 0; i < 24; ++i) {
                    const double rnow = p.norm();
                    if (rnow <= r_s) break;
                    sim::Vec2 inward = (rnow > 0.0) ? (-p / rnow) : sim::Vec2(-1.0, 0.0);
                    double stepLen = std::max(0.002, 0.05 * (rnow - r_s));
                    p += inward * stepLen;
                    rr.trail.push_back(p);
                }
            }
        }

        // Animate disk
        diskAngle = std::fmod(diskAngle + diskAngularSpeed * static_cast<float>(dt), 2.0f * PI);

        BeginDrawing();
        ClearBackground(BLACK);

        // Controls: adjust bending scale with [ and ]; zoom with -/=
        if (IsKeyPressed(KEY_LEFT_BRACKET)) {
            grav.bendingAngleScale = std::max(0.1, grav.bendingAngleScale * 0.8);
        }
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) {
            grav.bendingAngleScale = std::min(10.0, grav.bendingAngleScale * 1.25);
        }
        if (IsKeyPressed(KEY_MINUS)) {
            pixelsPerUnit = std::max(5.0f, pixelsPerUnit * 0.9f);
        }
        if (IsKeyPressed(KEY_EQUAL)) {
            pixelsPerUnit = std::min(120.0f, pixelsPerUnit * 1.1f);
        }

        // Stars with GR-like lensing: deflection ~ 4GM/(c^2 b).
        // Approximate a backward ray-map with a radial 1/r shift toward the BH.
        const double lensK = 0.5 * static_cast<double>(r_s); // tuning close to Einstein angle scale
        const double lensEps = 1e-6;
        for (const sim::Vec2& p : starsWorld) {
            const double r = p.norm();
            const sim::Vec2 rhat = (r > 0.0) ? (p / r) : sim::Vec2(1.0, 0.0);
            const double shift = lensK / std::max(r, lensEps);
            const sim::Vec2 lensed = p - shift * rhat;
            Vector2 sp = worldToScreen(lensed, screenW, screenH, pixelsPerUnit);
            DrawPixelV(sp, (Color){200, 200, 255, 255});
        }

        // Screen center
        const Vector2 screenCenter = {static_cast<float>(screenW) * 0.5f, static_cast<float>(screenH) * 0.5f};

        // Accretion disk with Doppler beaming and blue/red tint
        const int segs = 180;
        const double G = grav.gravitationalConstant;
        const double M = grav.blackHoleMass;
        const double c = grav.speedOfLight;
        const double rMid = 0.5 * (diskInner + diskOuter);
        const double v_orb = std::sqrt(std::max(0.0, (G * M) / std::max(rMid, 1e-6))); // Newtonian orbital speed
        const double beta = std::min(0.95, v_orb / std::max(c, 1e-9));
        const double gamma = 1.0 / std::sqrt(std::max(1e-9, 1.0 - beta * beta));
        const sim::Vec2 viewDir(-1.0, 0.0); // camera looks toward +x from -infinity
        for (int a = 0; a < 360; a += 2) {
            float ang = (a * DEG2RAD) + diskAngle;
            sim::Vec2 velDir(-std::sin(ang), std::cos(ang)); // tangential direction
            double cosTheta = velDir.dot(viewDir);
            double D = 1.0 / (gamma * std::max(1e-6, 1.0 - beta * cosTheta));
            double boost = std::pow(D, 3.0); // intensity beaming approx
            float brightness = static_cast<float>(std::clamp(0.45 + 0.55 * boost, 0.3, 3.0));
            // Hue: approaching -> blue (210 deg), receding -> red (0 deg)
            float hue = (cosTheta > 0.0) ? 210.0f : 0.0f;
            float sat = 0.85f;
            Color col = ColorFromHSV(hue, sat, std::min(1.0f, brightness));
            DrawRing(screenCenter,
                     diskInner * pixelsPerUnit,
                     diskOuter * pixelsPerUnit,
                     static_cast<float>(a), static_cast<float>(a + 2),
                     segs, col);
        }

        // Black hole event horizon and stylized blue/purple glow
        DrawCircleV(screenCenter, r_s * pixelsPerUnit, BLACK);
        // subtle blue glow just outside horizon
        DrawRing(screenCenter,
                 (r_s + 0.03f * r_s) * pixelsPerUnit,
                 (r_s + 0.20f * r_s) * pixelsPerUnit,
                 0, 360, 128, (Color){80, 120, 255, 70});
        // photon sphere highlight
        DrawRing(screenCenter,
                 (r_ph - 0.02f * r_s) * pixelsPerUnit,
                 (r_ph + 0.03f * r_s) * pixelsPerUnit,
                 0, 360, 128, (Color){180, 120, 255, 120});

        // Draw rays
        for (auto& rr : rays) {
            for (size_t i = 1; i < rr.trail.size(); ++i) {
                Vector2 a = worldToScreen(rr.trail[i - 1], screenW, screenH, pixelsPerUnit);
                Vector2 b = worldToScreen(rr.trail[i], screenW, screenH, pixelsPerUnit);
                Color lc = {(unsigned char)rr.color.r, (unsigned char)rr.color.g, (unsigned char)rr.color.b, 200};
                DrawLineEx(a, b, 2.0f, lc);
            }
            Vector2 head = worldToScreen(rr.ray.position, screenW, screenH, pixelsPerUnit);
            DrawCircleV(head, 4.0f, rr.color);
        }

        // Preview line while dragging
        if (isDragging) {
            Vector2 a = worldToScreen(dragStartWorld, screenW, screenH, pixelsPerUnit);
            Vector2 b = worldToScreen(mouseWorld, screenW, screenH, pixelsPerUnit);
            Color pc = previewPred >= 0 ? GREEN : RED; DrawLineV(a, b, pc); DrawCircleV(a, 4.0f, pc);
            DrawText(previewPred >= 0 ? "pred: escape" : "pred: capture", 20, 230, 20, pc);
        }

        DrawText("Blackhole Sim - GR Curvature", 20, 20, 20, RAYWHITE);
        DrawText(TextFormat("R_s=%.2f  b_c=%.2f  bend=%.2f  [/]  zoom -/+", r_s, (0.5f*3.0f*sqrtf(3.0f))*r_s, (float)grav.bendingAngleScale), 20, 110, 18, RAYWHITE);
        DrawText("Auto-spawned rays from ring; color = SVM prediction", 20, 50, 20, RAYWHITE);
        DrawText(TextFormat("dataset=%d  svm_acc=%.1f%%", (int)spawnedDataset.size(), svmAcc * 100.0f), 20, 80, 20, RAYWHITE);
        EndDrawing();
    }

    // No dataset collection or training on exit; SVM is trained at startup for interactivity

    CloseWindow();
    return 0;
}
