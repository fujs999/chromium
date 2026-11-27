// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifdef UNSAFE_BUFFERS_BUILD
// TODO(crbug.com/40285824): Remove this and spanify to fix the errors.
#pragma allow_unsafe_buffers
#endif

#include "media/gpu/v4l2/v4l2_vda_helpers.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>

#include "media/base/video_codecs.h"
#include "media/media_buildflags.h"
#include "testing/gtest/include/gtest/gtest.h"

#if BUILDFLAG(ENABLE_HEVC_PARSER_AND_HW_DECODER)
#include "media/parsers/h265_nalu_parser.h"
#endif  // BUILDFLAG(ENABLE_HEVC_PARSER_AND_HW_DECODER)

namespace media {

#if BUILDFLAG(ENABLE_HEVC_PARSER_AND_HW_DECODER)
namespace {

// Appends an Annex B NAL unit of |type| to |stream| and returns its size
// including the start code. |payload| must not contain start code emulation.
size_t AppendNalu(std::vector<uint8_t>& stream,
                  H265NALU::Type type,
                  const std::vector<uint8_t>& payload) {
  const size_t old_size = stream.size();
  // Start code.
  stream.insert(stream.end(), {0x00, 0x00, 0x00, 0x01});
  // NAL unit header: forbidden_zero_bit(1) = 0, nal_unit_type(6),
  // nuh_layer_id(6) = 0, nuh_temporal_id_plus1(3) = 1.
  stream.push_back(static_cast<uint8_t>(type << 1));
  stream.push_back(0x01);
  stream.insert(stream.end(), payload.begin(), payload.end());
  return stream.size() - old_size;
}

size_t AppendParameterSet(std::vector<uint8_t>& stream, H265NALU::Type type) {
  return AppendNalu(stream, type, {0xAA, 0xBB, 0xCC});
}

// Appends a slice segment. The first bit of the slice segment header is
// first_slice_segment_in_pic_flag.
size_t AppendSlice(std::vector<uint8_t>& stream,
                   H265NALU::Type type,
                   bool first_slice_segment_in_pic) {
  return AppendNalu(
      stream, type,
      {static_cast<uint8_t>(first_slice_segment_in_pic ? 0x80 : 0x20), 0xAA,
       0xBB, 0xCC});
}

}  // namespace

TEST(V4L2HEVCInputBufferFragmentSplitterTest, CreatedForHEVCProfiles) {
  auto splitter =
      v4l2_vda_helpers::InputBufferFragmentSplitter::CreateFromProfile(
          HEVCPROFILE_MAIN);
  ASSERT_TRUE(splitter);

  // Unlike the default splitter, the HEVC one keeps a complete access unit
  // pending until the start of the next one is seen.
  std::vector<uint8_t> stream;
  AppendSlice(stream, H265NALU::IDR_W_RADL, true);
  size_t endpos = 0;
  ASSERT_TRUE(
      splitter->AdvanceFrameFragment(stream.data(), stream.size(), &endpos));
  EXPECT_EQ(endpos, stream.size());
  EXPECT_TRUE(splitter->IsPartialFramePending());
}

TEST(V4L2HEVCInputBufferFragmentSplitterTest, OneAccessUnitPerBuffer) {
  v4l2_vda_helpers::HEVCInputBufferFragmentSplitter splitter;

  // The first buffer holds the parameter sets and the IDR picture.
  std::vector<uint8_t> key_frame;
  AppendParameterSet(key_frame, H265NALU::VPS_NUT);
  AppendParameterSet(key_frame, H265NALU::SPS_NUT);
  AppendParameterSet(key_frame, H265NALU::PPS_NUT);
  AppendSlice(key_frame, H265NALU::IDR_W_RADL, true);

  size_t endpos = 0;
  ASSERT_TRUE(splitter.AdvanceFrameFragment(key_frame.data(), key_frame.size(),
                                            &endpos));
  EXPECT_EQ(endpos, key_frame.size());
  EXPECT_TRUE(splitter.IsPartialFramePending());

  // The second buffer starts a new picture, which completes the pending one
  // without consuming any byte of the new buffer.
  std::vector<uint8_t> delta_frame;
  AppendSlice(delta_frame, H265NALU::TRAIL_R, true);

  ASSERT_TRUE(splitter.AdvanceFrameFragment(delta_frame.data(),
                                            delta_frame.size(), &endpos));
  EXPECT_EQ(endpos, 0u);
  EXPECT_FALSE(splitter.IsPartialFramePending());

  // Parsing the same buffer again consumes it as the new pending picture.
  ASSERT_TRUE(splitter.AdvanceFrameFragment(delta_frame.data(),
                                            delta_frame.size(), &endpos));
  EXPECT_EQ(endpos, delta_frame.size());
  EXPECT_TRUE(splitter.IsPartialFramePending());
}

TEST(V4L2HEVCInputBufferFragmentSplitterTest, MultipleSlicesStayTogether) {
  v4l2_vda_helpers::HEVCInputBufferFragmentSplitter splitter;

  std::vector<uint8_t> stream;
  size_t first_frame_size = AppendSlice(stream, H265NALU::IDR_W_RADL, true);
  first_frame_size += AppendSlice(stream, H265NALU::IDR_W_RADL, false);
  size_t second_frame_size = AppendSlice(stream, H265NALU::TRAIL_R, true);
  second_frame_size += AppendSlice(stream, H265NALU::TRAIL_R, false);
  ASSERT_EQ(stream.size(), first_frame_size + second_frame_size);

  size_t endpos = 0;
  ASSERT_TRUE(
      splitter.AdvanceFrameFragment(stream.data(), stream.size(), &endpos));
  EXPECT_EQ(endpos, first_frame_size);
  EXPECT_FALSE(splitter.IsPartialFramePending());

  ASSERT_TRUE(splitter.AdvanceFrameFragment(stream.data() + first_frame_size,
                                            second_frame_size, &endpos));
  EXPECT_EQ(endpos, second_frame_size);
  EXPECT_TRUE(splitter.IsPartialFramePending());
}

TEST(V4L2HEVCInputBufferFragmentSplitterTest,
     ParameterSetsAreGroupedWithFollowingFrame) {
  v4l2_vda_helpers::HEVCInputBufferFragmentSplitter splitter;

  std::vector<uint8_t> stream;
  const size_t first_frame_size = AppendSlice(stream, H265NALU::TRAIL_R, true);
  size_t second_frame_size = AppendParameterSet(stream, H265NALU::VPS_NUT);
  second_frame_size += AppendParameterSet(stream, H265NALU::SPS_NUT);
  second_frame_size += AppendParameterSet(stream, H265NALU::PPS_NUT);
  second_frame_size += AppendSlice(stream, H265NALU::IDR_W_RADL, true);

  size_t endpos = 0;
  ASSERT_TRUE(
      splitter.AdvanceFrameFragment(stream.data(), stream.size(), &endpos));
  EXPECT_EQ(endpos, first_frame_size);
  EXPECT_FALSE(splitter.IsPartialFramePending());

  ASSERT_TRUE(splitter.AdvanceFrameFragment(stream.data() + first_frame_size,
                                            second_frame_size, &endpos));
  EXPECT_EQ(endpos, second_frame_size);
  EXPECT_TRUE(splitter.IsPartialFramePending());
}

TEST(V4L2HEVCInputBufferFragmentSplitterTest, AccessUnitDelimiterSplitsFrames) {
  v4l2_vda_helpers::HEVCInputBufferFragmentSplitter splitter;

  std::vector<uint8_t> stream;
  size_t first_frame_size = AppendNalu(stream, H265NALU::AUD_NUT, {0x50});
  first_frame_size += AppendSlice(stream, H265NALU::IDR_W_RADL, true);
  size_t second_frame_size = AppendNalu(stream, H265NALU::AUD_NUT, {0x50});
  second_frame_size += AppendSlice(stream, H265NALU::TRAIL_R, true);

  size_t endpos = 0;
  ASSERT_TRUE(
      splitter.AdvanceFrameFragment(stream.data(), stream.size(), &endpos));
  EXPECT_EQ(endpos, first_frame_size);
  EXPECT_FALSE(splitter.IsPartialFramePending());

  ASSERT_TRUE(splitter.AdvanceFrameFragment(stream.data() + first_frame_size,
                                            second_frame_size, &endpos));
  EXPECT_EQ(endpos, second_frame_size);
  EXPECT_TRUE(splitter.IsPartialFramePending());
}

TEST(V4L2HEVCInputBufferFragmentSplitterTest, ResetDropsPendingFrame) {
  v4l2_vda_helpers::HEVCInputBufferFragmentSplitter splitter;

  std::vector<uint8_t> stream;
  AppendSlice(stream, H265NALU::IDR_W_RADL, true);

  size_t endpos = 0;
  ASSERT_TRUE(
      splitter.AdvanceFrameFragment(stream.data(), stream.size(), &endpos));
  EXPECT_TRUE(splitter.IsPartialFramePending());

  splitter.Reset();
  EXPECT_FALSE(splitter.IsPartialFramePending());

  // After a reset the next picture must not be seen as the end of a previous
  // one.
  ASSERT_TRUE(
      splitter.AdvanceFrameFragment(stream.data(), stream.size(), &endpos));
  EXPECT_EQ(endpos, stream.size());
  EXPECT_TRUE(splitter.IsPartialFramePending());
}

TEST(V4L2HEVCInputBufferFragmentSplitterTest, RejectsInvalidStream) {
  size_t endpos = 0;

  {
    // forbidden_zero_bit is set.
    v4l2_vda_helpers::HEVCInputBufferFragmentSplitter splitter;
    const std::vector<uint8_t> stream = {0x00, 0x00, 0x00, 0x01,
                                         0x82, 0x01, 0x80, 0xAA};
    EXPECT_FALSE(
        splitter.AdvanceFrameFragment(stream.data(), stream.size(), &endpos));
  }

  {
    // nuh_temporal_id_plus1 is zero.
    v4l2_vda_helpers::HEVCInputBufferFragmentSplitter splitter;
    const std::vector<uint8_t> stream = {0x00, 0x00, 0x00, 0x01,
                                         0x02, 0x00, 0x80, 0xAA};
    EXPECT_FALSE(
        splitter.AdvanceFrameFragment(stream.data(), stream.size(), &endpos));
  }

  {
    // A slice NAL unit without any slice segment header byte.
    v4l2_vda_helpers::HEVCInputBufferFragmentSplitter splitter;
    std::vector<uint8_t> stream;
    AppendNalu(stream, H265NALU::TRAIL_R, {});
    EXPECT_FALSE(
        splitter.AdvanceFrameFragment(stream.data(), stream.size(), &endpos));
  }
}
#endif  // BUILDFLAG(ENABLE_HEVC_PARSER_AND_HW_DECODER)

}  // namespace media
