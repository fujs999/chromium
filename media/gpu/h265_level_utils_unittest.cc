// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "media/gpu/h265_level_utils.h"

#include <optional>

#include "testing/gtest/include/gtest/gtest.h"
#include "ui/gfx/geometry/size.h"

namespace media {

namespace {
constexpr uint32_t kLowBitrate = 500000;
}  // namespace

TEST(H265LevelUtilsTest, PicksLowestLevelForCommonResolutions) {
  // 176x144 fits level 1 (MaxLumaPs 36864, MaxLumaSr 552960, MaxBR 128 kbps).
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(176, 144), 15, 100000), 30);
  // 640x360 exceeds the level 2 picture size (122880 samples).
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(640, 360), 30, kLowBitrate), 63);
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(1280, 720), 30, kLowBitrate), 93);
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(1920, 1080), 30, kLowBitrate), 120);
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(1920, 1088), 30, kLowBitrate), 120);
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(3840, 2160), 30, kLowBitrate), 150);
}

TEST(H265LevelUtilsTest, SampleRateRaisesLevel) {
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(1920, 1080), 60, kLowBitrate), 123);
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(3840, 2160), 60, kLowBitrate), 153);
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(3840, 2160), 120, kLowBitrate),
            156);
}

TEST(H265LevelUtilsTest, BitrateRaisesLevel) {
  // Level 3.1 allows 10 Mbps and level 4 allows 12 Mbps in the Main tier.
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(1280, 720), 30, 10000000), 93);
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(1280, 720), 30, 10000001), 120);
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(1280, 720), 30, 15000000), 123);
}

TEST(H265LevelUtilsTest, DimensionLimitRaisesLevel) {
  // 16384x16 is a small picture, but its width is only allowed from level 6
  // onwards: Sqrt(8 * 8912896) is about 8444, Sqrt(8 * 35651584) about 16888.
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(16384, 16), 30, kLowBitrate), 180);
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(16, 16384), 30, kLowBitrate), 180);
}

TEST(H265LevelUtilsTest, ZeroFramerateOnlyChecksPictureSize) {
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(1920, 1080), 0, kLowBitrate), 120);
}

TEST(H265LevelUtilsTest, RejectsUnsupportedConfigurations) {
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(), 30, kLowBitrate), std::nullopt);
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(0, 1080), 30, kLowBitrate),
            std::nullopt);
  // More luma samples than level 6.2 allows.
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(8192, 8192), 30, kLowBitrate),
            std::nullopt);
  // Higher sample rate than level 6.2 allows.
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(7680, 4320), 240, kLowBitrate),
            std::nullopt);
  // Higher bitrate than the Main tier of level 6.2 allows.
  EXPECT_EQ(FindValidH265LevelIdc(gfx::Size(1920, 1080), 30, 240000001),
            std::nullopt);
}

}  // namespace media
