#pragma once

#include <raylib.h>

// Minimal shim to allow using ImGui via raylib if user later adds Dear ImGui.
// For Step 1.1 we don't need full ImGui plumbing yet, but we keep a stable target.

inline void ImGui_ImplRaylib_Init() {}
inline void ImGui_ImplRaylib_NewFrame() {}
inline void ImGui_ImplRaylib_Render() {}
inline void ImGui_ImplRaylib_Shutdown() {}


