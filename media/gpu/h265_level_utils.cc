// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "media/gpu/h265_level_utils.h"

namespace media {

namespace {

struct H265LevelLimits {
  // general_level_idc, i.e. 30 times the level number.
  uint8_t level_idc;
  // Maximum luma picture size in samples (MaxLumaPs).
  uint64_t max_luma_ps;
  // Maximum luma sample rate in samples per second (MaxLumaSr).
  uint64_t max_luma_sr;
  // Maximum bitrate of the Main tier in units of 1000 bits per second (MaxBR).
  uint64_t max_br_kbps;
};

// Rec. ITU-T H.265 Table A.8 (general tier and level limits) and Table A.9
// (tier and level limits for the video profiles), Main tier only. Entries are
// sorted from the lowest to the highest level.
constexpr H265LevelLimits kH265MainTierLevelLimits[] = {
    {30, 36864, 552960, 128},             // Level 1
    {60, 122880, 3686400, 1500},          // Level 2
    {63, 245760, 7372800, 3000},          // Level 2.1
    {90, 552960, 16588800, 6000},         // Level 3
    {93, 983040, 33177600, 10000},        // Level 3.1
    {120, 2228224, 66846720, 12000},      // Level 4
    {123, 2228224, 133693440, 20000},     // Level 4.1
    {150, 8912896, 267386880, 25000},     // Level 5
    {153, 8912896, 534773760, 40000},     // Level 5.1
    {156, 8912896, 1069547520, 60000},    // Level 5.2
    {180, 35651584, 1069547520, 60000},   // Level 6
    {183, 35651584, 2139095040, 120000},  // Level 6.1
    {186, 35651584, 4278190080, 240000},  // Level 6.2
};

}  // namespace

std::optional<uint8_t> FindValidH265LevelIdc(const gfx::Size& coded_size,
                                             uint32_t framerate,
                                             uint32_t bitrate_bps) {
  if (coded_size.IsEmpty()) {
    return std::nullopt;
  }

  const uint64_t width = static_cast<uint64_t>(coded_size.width());
  const uint64_t height = static_cast<uint64_t>(coded_size.height());
  const uint64_t luma_ps = width * height;
  const uint64_t luma_sr = luma_ps * framerate;

  for (const auto& limits : kH265MainTierLevelLimits) {
    if (luma_ps > limits.max_luma_ps) {
      continue;
    }
    // Both dimensions must be less than or equal to Sqrt(8 * MaxLumaPs).
    if (width * width > 8 * limits.max_luma_ps ||
        height * height > 8 * limits.max_luma_ps) {
      continue;
    }
    if (luma_sr > limits.max_luma_sr) {
      continue;
    }
    if (bitrate_bps > limits.max_br_kbps * 1000) {
      continue;
    }
    return limits.level_idc;
  }

  return std::nullopt;
}

}  // namespace media
