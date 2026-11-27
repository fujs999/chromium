// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "media/gpu/vaapi/h265_vaapi_video_encoder_delegate.h"

#include <stdint.h>
#include <string.h>
#include <va/va.h>
#include <va/va_enc_hevc.h>

#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "base/compiler_specific.h"
#include "base/containers/span.h"
#include "base/functional/bind.h"
#include "base/functional/callback_helpers.h"
#include "base/memory/scoped_refptr.h"
#include "base/time/time.h"
#include "media/base/bitrate.h"
#include "media/base/video_bitrate_allocation.h"
#include "media/gpu/vaapi/vaapi_common.h"
#include "media/gpu/vaapi/vaapi_utils.h"
#include "media/gpu/vaapi/vaapi_wrapper.h"
#include "media/parsers/h265_parser.h"
#include "testing/gmock/include/gmock/gmock.h"
#include "testing/gtest/include/gtest/gtest.h"

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

namespace media {
namespace {

constexpr gfx::Size kDefaultVisibleSize(1280, 720);
constexpr uint32_t kDefaultBitrate = 2'000'000u;
constexpr uint32_t kDefaultFramerate = 30u;

// The level of a 1280x720 stream at 30 fps is 3.1 up to 10 Mbps and 4 above.
constexpr uint8_t kLevelIdc3_1 = 93;
constexpr uint8_t kLevelIdc4 = 120;

constexpr uint32_t kIDRPeriod = 2048;
constexpr uint32_t kCPBWindowSizeMs = 1500;

constexpr uint32_t kAllPackedHeaders =
    VA_ENC_PACKED_HEADER_SEQUENCE | VA_ENC_PACKED_HEADER_SLICE;

constexpr VaapiVideoEncoderDelegate::Config kDefaultVEADelegateConfig{
    .max_num_ref_frames = 1u | (1u << 16),
};

VideoEncodeAccelerator::Config DefaultVEAConfig() {
  return VideoEncodeAccelerator::Config(
      PIXEL_FORMAT_I420, kDefaultVisibleSize, HEVCPROFILE_MAIN,
      Bitrate::ConstantBitrate(kDefaultBitrate), kDefaultFramerate,
      VideoEncodeAccelerator::Config::StorageType::kShmem,
      VideoEncodeAccelerator::Config::ContentType::kCamera);
}

VideoBitrateAllocation CreateBitrateAllocation(uint32_t bitrate_bps) {
  VideoBitrateAllocation bitrate_allocation(Bitrate::Mode::kConstant);
  EXPECT_TRUE(bitrate_allocation.SetBitrate(0, 0, bitrate_bps));
  return bitrate_allocation;
}

// Returns the value of VAConfigAttribEncHEVCBlockSizes of a driver that
// supports the given block sizes.
uint32_t MakeBlockSizesAttribute(uint32_t log2_ctb_size,
                                 uint32_t log2_min_cb_size,
                                 uint32_t log2_max_tb_size,
                                 uint32_t log2_min_tb_size,
                                 uint32_t max_transform_hierarchy_depth) {
  VAConfigAttribValEncHEVCBlockSizes block_sizes;
  block_sizes.value = 0;
  block_sizes.bits.log2_max_coding_tree_block_size_minus3 = log2_ctb_size - 3;
  block_sizes.bits.log2_min_coding_tree_block_size_minus3 = log2_ctb_size - 3;
  block_sizes.bits.log2_min_luma_coding_block_size_minus3 =
      log2_min_cb_size - 3;
  block_sizes.bits.log2_max_luma_transform_block_size_minus2 =
      log2_max_tb_size - 2;
  block_sizes.bits.log2_min_luma_transform_block_size_minus2 =
      log2_min_tb_size - 2;
  block_sizes.bits.max_max_transform_hierarchy_depth_inter =
      max_transform_hierarchy_depth;
  block_sizes.bits.max_max_transform_hierarchy_depth_intra =
      max_transform_hierarchy_depth;
  return block_sizes.value;
}

// Returns the value of VAConfigAttribEncHEVCFeatures of a driver whose
// capabilities are similar to the ones of a hardware encoder that requires
// asymmetric motion partitions and CU QP delta, and supports sample adaptive
// offset, temporal motion vector prediction and transform skip.
uint32_t MakeFeaturesAttribute() {
  VAConfigAttribValEncHEVCFeatures features;
  features.value = 0;
  features.bits.amp = VA_FEATURE_REQUIRED;
  features.bits.sao = VA_FEATURE_SUPPORTED;
  features.bits.temporal_mvp = VA_FEATURE_SUPPORTED;
  features.bits.transform_skip = VA_FEATURE_SUPPORTED;
  features.bits.cu_qp_delta = VA_FEATURE_REQUIRED;
  return features.value;
}

// The syntax elements of a parsed slice segment header that are checked by
// the tests.
struct ParsedSliceHeader {
  int slice_type = 0;
  bool slice_temporal_mvp_enabled_flag = false;
  bool slice_sao_luma_flag = false;
  bool slice_sao_chroma_flag = false;
  bool mvd_l1_zero_flag = false;
};

// A copy of a buffer submitted to the driver.
struct SubmittedBuffer {
  VABufferType type;
  std::vector<uint8_t> data;
};

std::vector<VABufferType> GetBufferTypes(
    const std::vector<SubmittedBuffer>& buffers) {
  std::vector<VABufferType> types;
  for (const SubmittedBuffer& buffer : buffers) {
    types.push_back(buffer.type);
  }
  return types;
}

// Returns the structure of type |T| stored in |buffer|.
template <typename T>
T ReadBuffer(const SubmittedBuffer& buffer) {
  T value = {};
  EXPECT_EQ(buffer.data.size(), sizeof(T));
  if (buffer.data.size() == sizeof(T)) {
    UNSAFE_BUFFERS(memcpy(&value, buffer.data.data(), sizeof(T)));
  }
  return value;
}

// Returns the miscellaneous parameter of type |T| stored in |buffer|, whose
// type has to be |misc_param_type|.
template <typename T>
T ReadMiscBuffer(const SubmittedBuffer& buffer,
                 VAEncMiscParameterType misc_param_type) {
  T value = {};
  EXPECT_EQ(buffer.type, VAEncMiscParameterBufferType);
  EXPECT_EQ(buffer.data.size(), sizeof(VAEncMiscParameterBuffer) + sizeof(T));
  if (buffer.data.size() != sizeof(VAEncMiscParameterBuffer) + sizeof(T)) {
    return value;
  }
  VAEncMiscParameterBuffer header = {};
  UNSAFE_BUFFERS(memcpy(&header, buffer.data.data(), sizeof(header)));
  EXPECT_EQ(header.type, misc_param_type);
  base::span<const uint8_t> payload =
      base::span(buffer.data).subspan(sizeof(VAEncMiscParameterBuffer));
  UNSAFE_BUFFERS(memcpy(&value, payload.data(), sizeof(T)));
  return value;
}

bool IsInvalidVAPicture(const VAPictureHEVC& va_pic) {
  return va_pic.picture_id == VA_INVALID_ID &&
         va_pic.flags == VA_PICTURE_HEVC_INVALID;
}

bool AreSameVAPictures(const VAPictureHEVC& lhs, const VAPictureHEVC& rhs) {
  return lhs.picture_id == rhs.picture_id &&
         lhs.pic_order_cnt == rhs.pic_order_cnt && lhs.flags == rhs.flags;
}

// The buffer types submitted for a frame, in the order of submission.
const VABufferType kParameterBufferTypes[] = {
    VAEncSequenceParameterBufferType, VAEncPictureParameterBufferType,
    VAEncMiscParameterBufferType,     VAEncMiscParameterBufferType,
    VAEncMiscParameterBufferType,
};
const VABufferType kPackedHeaderBufferTypes[] = {
    VAEncPackedHeaderParameterBufferType,
    VAEncPackedHeaderDataBufferType,
};

std::vector<VABufferType> ExpectedBufferTypes(size_t num_packed_headers) {
  std::vector<VABufferType> types(std::begin(kParameterBufferTypes),
                                  std::end(kParameterBufferTypes));
  for (size_t i = 0; i < num_packed_headers; ++i) {
    types.insert(types.end(), std::begin(kPackedHeaderBufferTypes),
                 std::end(kPackedHeaderBufferTypes));
  }
  types.push_back(VAEncSliceParameterBufferType);
  return types;
}

class MockVaapiWrapper : public VaapiWrapper {
 public:
  MockVaapiWrapper()
      : VaapiWrapper(VADisplayStateHandle(), kEncodeConstantBitrate) {}

  MOCK_METHOD1(SubmitBuffer_Locked, bool(const VABufferDescriptor&));

  bool GetSupportedPackedHeaders(VideoCodecProfile profile,
                                 bool& packed_sps,
                                 bool& packed_pps,
                                 bool& packed_slice) override {
    packed_sps = packed_headers_ & VA_ENC_PACKED_HEADER_SEQUENCE;
    packed_pps = packed_headers_ & VA_ENC_PACKED_HEADER_PICTURE;
    packed_slice = packed_headers_ & VA_ENC_PACKED_HEADER_SLICE;
    return true;
  }

  bool GetEncConfigAttribute(VideoCodecProfile profile,
                             VAConfigAttribType type,
                             uint32_t* value) override {
    const auto it = attributes_.find(type);
    if (it == attributes_.end()) {
      return false;
    }
    *value = it->second;
    return true;
  }

  // Sets the packed headers supported by the driver.
  void set_packed_headers(uint32_t packed_headers) {
    packed_headers_ = packed_headers;
  }

  // Sets the value of the config attribute |type|. The attributes that are
  // not set are reported as not supported.
  void SetAttribute(VAConfigAttribType type, uint32_t value) {
    attributes_[type] = value;
  }

 protected:
  ~MockVaapiWrapper() override = default;

 private:
  uint32_t packed_headers_ = kAllPackedHeaders;
  std::map<VAConfigAttribType, uint32_t> attributes_;
};

}  // namespace

class H265VaapiVideoEncoderDelegateTest : public ::testing::Test {
 public:
  H265VaapiVideoEncoderDelegateTest() = default;

  void SetUp() override;

  MOCK_METHOD0(OnError, void());

 protected:
  // The parameters and the headers submitted to the driver for a frame.
  struct Frame {
    std::vector<SubmittedBuffer> buffers;
    VAEncSequenceParameterBufferHEVC seq_param;
    VAEncPictureParameterBufferHEVC pic_param;
    VAEncSliceParameterBufferHEVC slice_param;
    VAEncMiscParameterRateControl rate_control_param;
    VAEncMiscParameterFrameRate framerate_param;
    VAEncMiscParameterHRD hrd_param;
    // True if the frame is an IDR picture.
    bool is_idr = false;
    // The slice segment header parsed from the packed slice header, if any.
    std::optional<ParsedSliceHeader> slice_header;
  };

  // Recreates |encoder_| so that it can be initialized again.
  void CreateEncoder();

  bool InitializeEncoder(const VideoEncodeAccelerator::Config& config);

  // Encodes a frame. This verifies the buffers submitted to the driver: their
  // types, their order and that the packed headers are consistent with the
  // parameter buffers.
  Frame EncodeFrame(bool force_keyframe);

  bool UpdateRates(uint32_t bitrate_bps, uint32_t framerate);

  // Sets the packed headers supported by the driver. This has to be called
  // before InitializeEncoder().
  void SetSupportedPackedHeaders(uint32_t packed_headers) {
    packed_headers_ = packed_headers;
    mock_vaapi_wrapper_->set_packed_headers(packed_headers);
  }

  uint8_t level_idc() const { return encoder_->level_idc_; }

  // The parameter sets parsed from the last packed sequence header.
  const H265VPS* parsed_vps() const { return parser_.GetVPS(0); }
  const H265SPS* parsed_sps() const { return parser_.GetSPS(0); }
  const H265PPS* parsed_pps() const { return parser_.GetPPS(0); }

  scoped_refptr<MockVaapiWrapper> mock_vaapi_wrapper_;
  std::unique_ptr<H265VaapiVideoEncoderDelegate> encoder_;

 private:
  std::unique_ptr<VaapiVideoEncoderDelegate::EncodeJob> CreateEncodeJob(
      bool keyframe,
      base::TimeDelta timestamp);

  // Parses the packed header in |data| whose type is |packed_header_type| and
  // verifies it against the parameter buffers of |frame|.
  void VerifyPackedHeader(uint32_t packed_header_type,
                          const std::vector<uint8_t>& data,
                          Frame& frame);
  void VerifyParameterSets(const Frame& frame);
  void VerifySliceHeader(const H265SliceHeader& slice_header,
                         const Frame& frame);

  gfx::Size visible_size_;
  uint32_t packed_headers_ = kAllPackedHeaders;

  // The parser keeps the parameter sets of the last IDR picture, they are
  // needed to parse the slice segment headers.
  H265Parser parser_;

  unsigned int next_surface_id_ = 0;
  size_t num_encoded_frames_ = 0;
};

void H265VaapiVideoEncoderDelegateTest::SetUp() {
  mock_vaapi_wrapper_ = base::MakeRefCounted<MockVaapiWrapper>();
  ASSERT_TRUE(mock_vaapi_wrapper_);
  EXPECT_CALL(*this, OnError()).Times(0);
  CreateEncoder();
}

void H265VaapiVideoEncoderDelegateTest::CreateEncoder() {
  encoder_ = std::make_unique<H265VaapiVideoEncoderDelegate>(
      mock_vaapi_wrapper_,
      base::BindRepeating(&H265VaapiVideoEncoderDelegateTest::OnError,
                          base::Unretained(this)));
}

bool H265VaapiVideoEncoderDelegateTest::InitializeEncoder(
    const VideoEncodeAccelerator::Config& config) {
  visible_size_ = config.input_visible_size;
  return encoder_->Initialize(config, kDefaultVEADelegateConfig);
}

std::unique_ptr<VaapiVideoEncoderDelegate::EncodeJob>
H265VaapiVideoEncoderDelegateTest::CreateEncodeJob(bool keyframe,
                                                   base::TimeDelta timestamp) {
  auto picture = base::MakeRefCounted<VaapiH265Picture>(
      std::make_unique<VASurfaceHandle>(next_surface_id_++, base::DoNothing()));

  constexpr VABufferID kDummyVABufferID = 12;
  auto scoped_va_buffer = ScopedVABuffer::CreateForTesting(
      kDummyVABufferID, VAEncCodedBufferType, visible_size_.GetArea());

  return std::make_unique<VaapiVideoEncoderDelegate::EncodeJob>(
      keyframe, timestamp, /*spatial_index=*/0u, /*end_of_picture=*/true,
      next_surface_id_++, picture, std::move(scoped_va_buffer));
}

H265VaapiVideoEncoderDelegateTest::Frame
H265VaapiVideoEncoderDelegateTest::EncodeFrame(bool force_keyframe) {
  const base::TimeDelta timestamp =
      base::Milliseconds(33) * base::checked_cast<int>(num_encoded_frames_++);
  auto encode_job = CreateEncodeJob(force_keyframe, timestamp);

  Frame frame;
  EXPECT_CALL(*mock_vaapi_wrapper_, SubmitBuffer_Locked(_))
      .WillRepeatedly([&frame](
                          const VaapiWrapper::VABufferDescriptor& va_buffer) {
        EXPECT_TRUE(va_buffer.data);
        // SAFETY: The delegate guarantees that |va_buffer.data| points to
        // |va_buffer.size| bytes.
        auto data = UNSAFE_BUFFERS(base::span(
            static_cast<const uint8_t*>(va_buffer.data.get()), va_buffer.size));
        frame.buffers.push_back(
            {va_buffer.type, std::vector<uint8_t>(data.begin(), data.end())});
        return true;
      });
  EXPECT_EQ(encoder_->PrepareEncodeJob(*encode_job),
            VaapiVideoEncoderDelegate::PrepareEncodeJobResult::kSuccess);
  ::testing::Mock::VerifyAndClearExpectations(mock_vaapi_wrapper_.get());

  // The parameter buffers are followed by the packed headers and the slice
  // parameter buffer is the last one.
  const std::vector<SubmittedBuffer>& buffers = frame.buffers;
  if (buffers.size() < std::size(kParameterBufferTypes) + 1) {
    ADD_FAILURE() << "Too few buffers are submitted: " << buffers.size();
    return frame;
  }
  frame.seq_param = ReadBuffer<VAEncSequenceParameterBufferHEVC>(buffers[0]);
  frame.pic_param = ReadBuffer<VAEncPictureParameterBufferHEVC>(buffers[1]);
  frame.rate_control_param = ReadMiscBuffer<VAEncMiscParameterRateControl>(
      buffers[2], VAEncMiscParameterTypeRateControl);
  frame.framerate_param = ReadMiscBuffer<VAEncMiscParameterFrameRate>(
      buffers[3], VAEncMiscParameterTypeFrameRate);
  frame.hrd_param = ReadMiscBuffer<VAEncMiscParameterHRD>(
      buffers[4], VAEncMiscParameterTypeHRD);
  frame.slice_param = ReadBuffer<VAEncSliceParameterBufferHEVC>(buffers.back());
  frame.is_idr = frame.pic_param.pic_fields.bits.idr_pic_flag;

  // An IDR picture is produced whenever a keyframe is requested.
  if (force_keyframe) {
    EXPECT_TRUE(frame.is_idr);
  }

  const bool has_packed_parameter_sets =
      packed_headers_ & VA_ENC_PACKED_HEADER_SEQUENCE;
  const bool has_packed_slice_header =
      has_packed_parameter_sets &&
      (packed_headers_ & VA_ENC_PACKED_HEADER_SLICE);
  std::vector<uint32_t> expected_packed_header_types;
  if (has_packed_parameter_sets && frame.is_idr) {
    expected_packed_header_types.push_back(VAEncPackedHeaderSequence);
  }
  if (has_packed_slice_header) {
    expected_packed_header_types.push_back(VAEncPackedHeaderSlice);
  }
  EXPECT_THAT(GetBufferTypes(buffers),
              ElementsAreArray(
                  ExpectedBufferTypes(expected_packed_header_types.size())));
  if (GetBufferTypes(buffers) !=
      ExpectedBufferTypes(expected_packed_header_types.size())) {
    return frame;
  }

  // Verify the packed headers.
  for (size_t i = 0; i < expected_packed_header_types.size(); ++i) {
    const size_t index = std::size(kParameterBufferTypes) + 2 * i;
    const auto packed_header_param =
        ReadBuffer<VAEncPackedHeaderParameterBuffer>(buffers[index]);
    const std::vector<uint8_t>& packed_header_data = buffers[index + 1].data;
    EXPECT_EQ(packed_header_param.type, expected_packed_header_types[i]);
    EXPECT_EQ(packed_header_param.bit_length, packed_header_data.size() * 8);
    // The packed headers contain emulation prevention bytes.
    EXPECT_EQ(packed_header_param.has_emulation_bytes, 1);
    VerifyPackedHeader(packed_header_param.type, packed_header_data, frame);
  }

  // Verify the metadata of the encoded frame.
  constexpr size_t kDummyPayloadSize = 12345;
  const BitstreamBufferMetadata metadata =
      encoder_->GetMetadata(*encode_job, kDummyPayloadSize);
  EXPECT_EQ(metadata.payload_size_bytes, kDummyPayloadSize);
  EXPECT_EQ(metadata.key_frame, frame.is_idr);
  EXPECT_EQ(metadata.timestamp, timestamp);
  // The QP is unknown because the rate control is done by the driver.
  EXPECT_EQ(metadata.qp, -1);

  return frame;
}

void H265VaapiVideoEncoderDelegateTest::VerifyPackedHeader(
    uint32_t packed_header_type,
    const std::vector<uint8_t>& data,
    Frame& frame) {
  parser_.SetStream(data.data(), base::checked_cast<off_t>(data.size()));

  std::vector<int> nal_unit_types;
  H265NALU nalu;
  while (true) {
    const H265Parser::Result result = parser_.AdvanceToNextNALU(&nalu);
    if (result == H265Parser::kEOStream) {
      break;
    }
    ASSERT_EQ(result, H265Parser::kOk);
    nal_unit_types.push_back(nalu.nal_unit_type);
    // The NALUs are in the base layer and in the lowest temporal sub-layer.
    EXPECT_EQ(nalu.nuh_layer_id, 0);
    EXPECT_EQ(nalu.nuh_temporal_id_plus1, 1);

    int id = -1;
    switch (nalu.nal_unit_type) {
      case H265NALU::VPS_NUT:
        ASSERT_EQ(parser_.ParseVPS(&id), H265Parser::kOk);
        EXPECT_EQ(id, 0);
        break;
      case H265NALU::SPS_NUT:
        ASSERT_EQ(parser_.ParseSPS(&id), H265Parser::kOk);
        EXPECT_EQ(id, 0);
        break;
      case H265NALU::PPS_NUT:
        ASSERT_EQ(parser_.ParsePPS(nalu, &id), H265Parser::kOk);
        EXPECT_EQ(id, 0);
        break;
      case H265NALU::IDR_W_RADL:
      case H265NALU::TRAIL_R: {
        H265SliceHeader slice_header;
        ASSERT_EQ(parser_.ParseSliceHeader(nalu, &slice_header, nullptr),
                  H265Parser::kOk);
        // The packed slice header only consists of the slice segment header,
        // which ends with byte_alignment().
        EXPECT_EQ(slice_header.header_size +
                      slice_header.header_emulation_prevention_bytes,
                  slice_header.nalu_size);
        VerifySliceHeader(slice_header, frame);
        frame.slice_header = ParsedSliceHeader{
            .slice_type = slice_header.slice_type,
            .slice_temporal_mvp_enabled_flag =
                slice_header.slice_temporal_mvp_enabled_flag,
            .slice_sao_luma_flag = slice_header.slice_sao_luma_flag,
            .slice_sao_chroma_flag = slice_header.slice_sao_chroma_flag,
            .mvd_l1_zero_flag = slice_header.mvd_l1_zero_flag,
        };
        break;
      }
      default:
        FAIL() << "Unexpected NALU type: " << nalu.nal_unit_type;
    }
  }

  if (packed_header_type == VAEncPackedHeaderSequence) {
    // All the parameter sets are submitted in a packed sequence header.
    EXPECT_THAT(
        nal_unit_types,
        ElementsAre(H265NALU::VPS_NUT, H265NALU::SPS_NUT, H265NALU::PPS_NUT));
    VerifyParameterSets(frame);
  } else {
    ASSERT_EQ(packed_header_type,
              static_cast<uint32_t>(VAEncPackedHeaderSlice));
    EXPECT_THAT(nal_unit_types, ElementsAre(frame.pic_param.nal_unit_type));
  }
}

void H265VaapiVideoEncoderDelegateTest::VerifyParameterSets(
    const Frame& frame) {
  const H265VPS* vps = parsed_vps();
  const H265SPS* sps = parsed_sps();
  const H265PPS* pps = parsed_pps();
  ASSERT_TRUE(vps);
  ASSERT_TRUE(sps);
  ASSERT_TRUE(pps);

  const VAEncSequenceParameterBufferHEVC& seq_param = frame.seq_param;
  const VAEncPictureParameterBufferHEVC& pic_param = frame.pic_param;

  // Video parameter set.
  EXPECT_TRUE(vps->vps_base_layer_internal_flag);
  EXPECT_TRUE(vps->vps_base_layer_available_flag);
  EXPECT_EQ(vps->vps_max_layers_minus1, 0);
  EXPECT_EQ(vps->vps_max_sub_layers_minus1, 0);
  EXPECT_TRUE(vps->vps_temporal_id_nesting_flag);
  EXPECT_EQ(vps->profile_tier_level.general_profile_idc,
            H265ProfileTierLevel::kProfileIdcMain);
  EXPECT_EQ(vps->profile_tier_level.general_profile_idc,
            seq_param.general_profile_idc);
  EXPECT_EQ(vps->profile_tier_level.general_level_idc,
            seq_param.general_level_idc);
  EXPECT_TRUE(vps->profile_tier_level.general_progressive_source_flag);
  EXPECT_FALSE(vps->profile_tier_level.general_interlaced_source_flag);
  EXPECT_TRUE(vps->profile_tier_level.general_frame_only_constraint_flag);
  // A picture and its reference picture are kept, without any reordering.
  EXPECT_EQ(vps->vps_max_dec_pic_buffering_minus1[0], 1);
  EXPECT_EQ(vps->vps_max_num_reorder_pics[0], 0);

  // Sequence parameter set.
  EXPECT_EQ(sps->sps_video_parameter_set_id, vps->vps_video_parameter_set_id);
  EXPECT_EQ(sps->sps_max_sub_layers_minus1, 0);
  EXPECT_EQ(sps->profile_tier_level.general_profile_idc,
            seq_param.general_profile_idc);
  EXPECT_EQ(sps->profile_tier_level.general_level_idc,
            seq_param.general_level_idc);
  EXPECT_EQ(sps->profile_tier_level.general_level_idc, level_idc());
  EXPECT_EQ(sps->chroma_format_idc,
            static_cast<int>(seq_param.seq_fields.bits.chroma_format_idc));
  EXPECT_EQ(sps->chroma_format_idc, 1);
  EXPECT_EQ(sps->pic_width_in_luma_samples,
            seq_param.pic_width_in_luma_samples);
  EXPECT_EQ(sps->pic_height_in_luma_samples,
            seq_param.pic_height_in_luma_samples);
  EXPECT_EQ(sps->GetCodedSize(), encoder_->GetCodedSize());
  EXPECT_EQ(sps->GetVisibleRect(), gfx::Rect(visible_size_));
  EXPECT_EQ(sps->bit_depth_luma_minus8, 0);
  EXPECT_EQ(sps->bit_depth_chroma_minus8, 0);
  // The picture order count LSB doesn't wrap in an IDR period.
  EXPECT_GE(static_cast<uint32_t>(sps->max_pic_order_cnt_lsb), kIDRPeriod);
  EXPECT_EQ(sps->sps_max_dec_pic_buffering_minus1[0], 1);
  EXPECT_EQ(sps->sps_max_num_reorder_pics[0], 0);
  EXPECT_EQ(sps->log2_min_luma_coding_block_size_minus3,
            seq_param.log2_min_luma_coding_block_size_minus3);
  EXPECT_EQ(sps->log2_diff_max_min_luma_coding_block_size,
            seq_param.log2_diff_max_min_luma_coding_block_size);
  EXPECT_EQ(sps->log2_min_luma_transform_block_size_minus2,
            seq_param.log2_min_transform_block_size_minus2);
  EXPECT_EQ(sps->log2_diff_max_min_luma_transform_block_size,
            seq_param.log2_diff_max_min_transform_block_size);
  EXPECT_EQ(sps->max_transform_hierarchy_depth_inter,
            seq_param.max_transform_hierarchy_depth_inter);
  EXPECT_EQ(sps->max_transform_hierarchy_depth_intra,
            seq_param.max_transform_hierarchy_depth_intra);
  EXPECT_FALSE(sps->scaling_list_enabled_flag);
  EXPECT_FALSE(seq_param.seq_fields.bits.scaling_list_enabled_flag);
  EXPECT_EQ(sps->amp_enabled_flag,
            !!seq_param.seq_fields.bits.amp_enabled_flag);
  EXPECT_EQ(sps->sample_adaptive_offset_enabled_flag,
            !!seq_param.seq_fields.bits.sample_adaptive_offset_enabled_flag);
  EXPECT_FALSE(sps->pcm_enabled_flag);
  EXPECT_FALSE(seq_param.seq_fields.bits.pcm_enabled_flag);
  // The reference picture sets are signaled in the slice segment headers.
  EXPECT_EQ(sps->num_short_term_ref_pic_sets, 0);
  EXPECT_FALSE(sps->long_term_ref_pics_present_flag);
  EXPECT_EQ(sps->sps_temporal_mvp_enabled_flag,
            !!seq_param.seq_fields.bits.sps_temporal_mvp_enabled_flag);
  EXPECT_EQ(sps->strong_intra_smoothing_enabled_flag,
            !!seq_param.seq_fields.bits.strong_intra_smoothing_enabled_flag);
  EXPECT_FALSE(seq_param.vui_parameters_present_flag);
  // The whole picture is a single slice.
  EXPECT_EQ(static_cast<uint32_t>(sps->pic_size_in_ctbs_y),
            frame.slice_param.num_ctu_in_slice);

  // Picture parameter set.
  EXPECT_EQ(pps->pps_pic_parameter_set_id,
            pic_param.slice_pic_parameter_set_id);
  EXPECT_EQ(pps->pps_seq_parameter_set_id, sps->sps_seq_parameter_set_id);
  EXPECT_EQ(pps->dependent_slice_segments_enabled_flag,
            !!pic_param.pic_fields.bits.dependent_slice_segments_enabled_flag);
  EXPECT_FALSE(pps->output_flag_present_flag);
  EXPECT_EQ(pps->num_extra_slice_header_bits, 0);
  EXPECT_EQ(pps->sign_data_hiding_enabled_flag,
            !!pic_param.pic_fields.bits.sign_data_hiding_enabled_flag);
  EXPECT_FALSE(pps->cabac_init_present_flag);
  EXPECT_EQ(pps->num_ref_idx_l0_default_active_minus1,
            pic_param.num_ref_idx_l0_default_active_minus1);
  EXPECT_EQ(pps->num_ref_idx_l1_default_active_minus1,
            pic_param.num_ref_idx_l1_default_active_minus1);
  EXPECT_EQ(pps->init_qp_minus26 + 26, pic_param.pic_init_qp);
  EXPECT_EQ(pps->constrained_intra_pred_flag,
            !!pic_param.pic_fields.bits.constrained_intra_pred_flag);
  EXPECT_EQ(pps->transform_skip_enabled_flag,
            !!pic_param.pic_fields.bits.transform_skip_enabled_flag);
  EXPECT_EQ(pps->cu_qp_delta_enabled_flag,
            !!pic_param.pic_fields.bits.cu_qp_delta_enabled_flag);
  EXPECT_EQ(pps->diff_cu_qp_delta_depth, pic_param.diff_cu_qp_delta_depth);
  EXPECT_EQ(pps->pps_cb_qp_offset, pic_param.pps_cb_qp_offset);
  EXPECT_EQ(pps->pps_cr_qp_offset, pic_param.pps_cr_qp_offset);
  EXPECT_FALSE(pps->pps_slice_chroma_qp_offsets_present_flag);
  EXPECT_EQ(pps->weighted_pred_flag,
            !!pic_param.pic_fields.bits.weighted_pred_flag);
  EXPECT_EQ(pps->weighted_bipred_flag,
            !!pic_param.pic_fields.bits.weighted_bipred_flag);
  EXPECT_EQ(pps->transquant_bypass_enabled_flag,
            !!pic_param.pic_fields.bits.transquant_bypass_enabled_flag);
  EXPECT_EQ(pps->tiles_enabled_flag,
            !!pic_param.pic_fields.bits.tiles_enabled_flag);
  EXPECT_EQ(pps->entropy_coding_sync_enabled_flag,
            !!pic_param.pic_fields.bits.entropy_coding_sync_enabled_flag);
  EXPECT_EQ(
      pps->pps_loop_filter_across_slices_enabled_flag,
      !!pic_param.pic_fields.bits.pps_loop_filter_across_slices_enabled_flag);
  EXPECT_FALSE(pps->pps_scaling_list_data_present_flag);
  EXPECT_FALSE(pic_param.pic_fields.bits.scaling_list_data_present_flag);
  EXPECT_FALSE(pps->lists_modification_present_flag);
  EXPECT_EQ(pps->log2_parallel_merge_level_minus2,
            pic_param.log2_parallel_merge_level_minus2);
}

void H265VaapiVideoEncoderDelegateTest::VerifySliceHeader(
    const H265SliceHeader& slice_header,
    const Frame& frame) {
  const VAEncPictureParameterBufferHEVC& pic_param = frame.pic_param;
  const VAEncSliceParameterBufferHEVC& slice_param = frame.slice_param;

  EXPECT_EQ(slice_header.nal_unit_type, pic_param.nal_unit_type);
  EXPECT_EQ(slice_header.irap_pic, frame.is_idr);
  EXPECT_TRUE(slice_header.first_slice_segment_in_pic_flag);
  EXPECT_FALSE(slice_header.no_output_of_prior_pics_flag);
  EXPECT_EQ(slice_header.slice_pic_parameter_set_id,
            slice_param.slice_pic_parameter_set_id);
  EXPECT_FALSE(slice_header.dependent_slice_segment_flag);
  EXPECT_EQ(slice_header.slice_type, slice_param.slice_type);
  EXPECT_EQ(slice_header.slice_sao_luma_flag,
            !!slice_param.slice_fields.bits.slice_sao_luma_flag);
  EXPECT_EQ(slice_header.slice_sao_chroma_flag,
            !!slice_param.slice_fields.bits.slice_sao_chroma_flag);
  EXPECT_EQ(slice_header.slice_temporal_mvp_enabled_flag,
            !!slice_param.slice_fields.bits.slice_temporal_mvp_enabled_flag);
  EXPECT_EQ(slice_header.slice_qp_delta, slice_param.slice_qp_delta);
  EXPECT_EQ(slice_header.slice_cb_qp_offset, slice_param.slice_cb_qp_offset);
  EXPECT_EQ(slice_header.slice_cr_qp_offset, slice_param.slice_cr_qp_offset);
  EXPECT_EQ(
      slice_header.slice_deblocking_filter_disabled_flag,
      !!slice_param.slice_fields.bits.slice_deblocking_filter_disabled_flag);
  EXPECT_EQ(slice_header.slice_loop_filter_across_slices_enabled_flag,
            !!slice_param.slice_fields.bits
                  .slice_loop_filter_across_slices_enabled_flag);

  if (frame.is_idr) {
    EXPECT_TRUE(slice_header.IsISlice());
    return;
  }

  // An inter picture references the previous picture only, which is signaled
  // by a reference picture set in the slice segment header.
  EXPECT_FALSE(slice_header.IsISlice());
  const VAPictureHEVC& curr_pic = pic_param.decoded_curr_pic;
  const VAPictureHEVC& ref_pic = slice_param.ref_pic_list0[0];
  EXPECT_EQ(slice_header.slice_pic_order_cnt_lsb, curr_pic.pic_order_cnt);
  EXPECT_FALSE(slice_header.short_term_ref_pic_set_sps_flag);
  const H265StRefPicSet& st_ref_pic_set = slice_header.st_ref_pic_set;
  EXPECT_EQ(st_ref_pic_set.num_negative_pics, 1);
  EXPECT_EQ(st_ref_pic_set.num_positive_pics, 0);
  EXPECT_EQ(st_ref_pic_set.delta_poc_s0[0],
            ref_pic.pic_order_cnt - curr_pic.pic_order_cnt);
  EXPECT_EQ(st_ref_pic_set.used_by_curr_pic_s0[0], 1);
  EXPECT_EQ(slice_header.num_pic_total_curr, 1);
  EXPECT_EQ(slice_header.num_long_term_pics, 0);

  EXPECT_EQ(slice_header.num_ref_idx_active_override_flag,
            !!slice_param.slice_fields.bits.num_ref_idx_active_override_flag);
  EXPECT_EQ(slice_header.num_ref_idx_l0_active_minus1,
            slice_param.num_ref_idx_l0_active_minus1);
  EXPECT_EQ(slice_header.num_ref_idx_l1_active_minus1,
            slice_param.num_ref_idx_l1_active_minus1);
  EXPECT_EQ(slice_header.mvd_l1_zero_flag,
            !!slice_param.slice_fields.bits.mvd_l1_zero_flag);
  EXPECT_EQ(slice_header.cabac_init_flag,
            !!slice_param.slice_fields.bits.cabac_init_flag);
  EXPECT_EQ(slice_header.collocated_from_l0_flag,
            !!slice_param.slice_fields.bits.collocated_from_l0_flag);
  EXPECT_EQ(slice_header.collocated_ref_idx, 0);
  EXPECT_EQ(5 - slice_header.five_minus_max_num_merge_cand,
            slice_param.max_num_merge_cand);
}

bool H265VaapiVideoEncoderDelegateTest::UpdateRates(uint32_t bitrate_bps,
                                                    uint32_t framerate) {
  return encoder_->UpdateRates(CreateBitrateAllocation(bitrate_bps), framerate);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, Initialize) {
  EXPECT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  // The driver doesn't report the block sizes, so the minimum coding block
  // size is 16x16.
  EXPECT_EQ(encoder_->GetCodedSize(), kDefaultVisibleSize);
  EXPECT_EQ(encoder_->GetMaxNumOfRefFrames(), 1u);
  EXPECT_THAT(encoder_->GetSVCLayerResolutions(),
              ElementsAre(kDefaultVisibleSize));
  EXPECT_EQ(level_idc(), kLevelIdc3_1);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, InitializeFailsForUnsupportedConfig) {
  {
    // Only the Main profile is supported.
    auto config = DefaultVEAConfig();
    config.output_profile = HEVCPROFILE_MAIN10;
    EXPECT_FALSE(InitializeEncoder(config));
  }
  {
    auto config = DefaultVEAConfig();
    config.output_profile = H264PROFILE_MAIN;
    CreateEncoder();
    EXPECT_FALSE(InitializeEncoder(config));
  }
  {
    // The picture sizes have to be even.
    auto config = DefaultVEAConfig();
    config.input_visible_size = gfx::Size(1281, 720);
    CreateEncoder();
    EXPECT_FALSE(InitializeEncoder(config));
  }
  {
    auto config = DefaultVEAConfig();
    config.input_visible_size = gfx::Size(1280, 721);
    CreateEncoder();
    EXPECT_FALSE(InitializeEncoder(config));
  }
  {
    auto config = DefaultVEAConfig();
    config.input_visible_size = gfx::Size();
    CreateEncoder();
    EXPECT_FALSE(InitializeEncoder(config));
  }
  {
    // The picture size exceeds the limit of the highest level.
    auto config = DefaultVEAConfig();
    config.input_visible_size = gfx::Size(8192, 8192);
    CreateEncoder();
    EXPECT_FALSE(InitializeEncoder(config));
  }
  {
    // Only the constant bitrate mode is supported.
    auto config = DefaultVEAConfig();
    config.bitrate =
        Bitrate::VariableBitrate(kDefaultBitrate, 2u * kDefaultBitrate);
    CreateEncoder();
    EXPECT_FALSE(InitializeEncoder(config));
  }
  {
    // Temporal layers are not supported.
    auto config = DefaultVEAConfig();
    config.spatial_layers.resize(1u);
    auto& spatial_layer = config.spatial_layers[0];
    spatial_layer.width = config.input_visible_size.width();
    spatial_layer.height = config.input_visible_size.height();
    spatial_layer.bitrate_bps = config.bitrate.target_bps();
    spatial_layer.framerate = config.framerate;
    spatial_layer.max_qp = 30;
    spatial_layer.num_of_temporal_layers = 2;
    CreateEncoder();
    EXPECT_FALSE(InitializeEncoder(config));
  }
  {
    // Spatial layers are not supported.
    auto config = DefaultVEAConfig();
    config.spatial_layers.resize(2u);
    for (size_t i = 0; i < config.spatial_layers.size(); ++i) {
      auto& spatial_layer = config.spatial_layers[i];
      const int divisor = i == 0 ? 2 : 1;
      spatial_layer.width = config.input_visible_size.width() / divisor;
      spatial_layer.height = config.input_visible_size.height() / divisor;
      spatial_layer.bitrate_bps = config.bitrate.target_bps() / 2;
      spatial_layer.framerate = config.framerate;
      spatial_layer.max_qp = 30;
      spatial_layer.num_of_temporal_layers = 1;
    }
    CreateEncoder();
    EXPECT_FALSE(InitializeEncoder(config));
  }
  {
    // The default config is supported by a recreated encoder.
    CreateEncoder();
    EXPECT_TRUE(InitializeEncoder(DefaultVEAConfig()));
  }
}

TEST_F(H265VaapiVideoEncoderDelegateTest,
       CodedSizeIsAlignedToDefaultMinimumCodingBlockSize) {
  auto config = DefaultVEAConfig();
  config.input_visible_size = gfx::Size(640, 360);
  ASSERT_TRUE(InitializeEncoder(config));

  // The driver doesn't report the block sizes, so the coding tree block size
  // is 32x32 and the minimum coding block size is 16x16.
  EXPECT_EQ(encoder_->GetCodedSize(), gfx::Size(640, 368));

  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_EQ(frame.seq_param.pic_width_in_luma_samples, 640);
  EXPECT_EQ(frame.seq_param.pic_height_in_luma_samples, 368);
  EXPECT_EQ(frame.seq_param.log2_min_luma_coding_block_size_minus3, 1);
  EXPECT_EQ(frame.seq_param.log2_diff_max_min_luma_coding_block_size, 1);
  EXPECT_EQ(frame.seq_param.log2_min_transform_block_size_minus2, 0);
  EXPECT_EQ(frame.seq_param.log2_diff_max_min_transform_block_size, 3);
  EXPECT_EQ(frame.seq_param.max_transform_hierarchy_depth_inter, 3);
  EXPECT_EQ(frame.seq_param.max_transform_hierarchy_depth_intra, 3);
  // 20 x 12 coding tree blocks of 32x32.
  EXPECT_EQ(frame.slice_param.num_ctu_in_slice, 240u);

  // The padding is cropped by the conformance window, which is verified by
  // EncodeFrame(). Its offsets are in units of chroma samples.
  const H265SPS* sps = parsed_sps();
  ASSERT_TRUE(sps);
  EXPECT_EQ(sps->conf_win_left_offset, 0);
  EXPECT_EQ(sps->conf_win_right_offset, 0);
  EXPECT_EQ(sps->conf_win_top_offset, 0);
  EXPECT_EQ(sps->conf_win_bottom_offset, 4);
}

TEST_F(H265VaapiVideoEncoderDelegateTest,
       CodedSizeIsAlignedToReportedMinimumCodingBlockSize) {
  // 64x64 coding tree blocks, 8x8 minimum coding blocks and 4x4 to 32x32
  // transform blocks.
  mock_vaapi_wrapper_->SetAttribute(
      VAConfigAttribEncHEVCBlockSizes,
      MakeBlockSizesAttribute(/*log2_ctb_size=*/6, /*log2_min_cb_size=*/3,
                              /*log2_max_tb_size=*/5, /*log2_min_tb_size=*/2,
                              /*max_transform_hierarchy_depth=*/2));

  auto config = DefaultVEAConfig();
  config.input_visible_size = gfx::Size(640, 360);
  ASSERT_TRUE(InitializeEncoder(config));
  EXPECT_EQ(encoder_->GetCodedSize(), gfx::Size(640, 360));

  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_EQ(frame.seq_param.pic_width_in_luma_samples, 640);
  EXPECT_EQ(frame.seq_param.pic_height_in_luma_samples, 360);
  EXPECT_EQ(frame.seq_param.log2_min_luma_coding_block_size_minus3, 0);
  EXPECT_EQ(frame.seq_param.log2_diff_max_min_luma_coding_block_size, 3);
  EXPECT_EQ(frame.seq_param.log2_min_transform_block_size_minus2, 0);
  EXPECT_EQ(frame.seq_param.log2_diff_max_min_transform_block_size, 3);
  EXPECT_EQ(frame.seq_param.max_transform_hierarchy_depth_inter, 2);
  EXPECT_EQ(frame.seq_param.max_transform_hierarchy_depth_intra, 2);
  // 10 x 6 coding tree blocks of 64x64.
  EXPECT_EQ(frame.slice_param.num_ctu_in_slice, 60u);

  const H265SPS* sps = parsed_sps();
  ASSERT_TRUE(sps);
  EXPECT_EQ(sps->conf_win_right_offset, 0);
  EXPECT_EQ(sps->conf_win_bottom_offset, 0);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, UnexpectedBlockSizesAreClamped) {
  // The driver reports 8x8 coding tree blocks and 4x4 maximum transform
  // blocks, which don't conform to the specification.
  mock_vaapi_wrapper_->SetAttribute(VAConfigAttribEncHEVCBlockSizes, 0u);
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  // EncodeFrame() verifies that the parameter sets are valid.
  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  // 16x16 coding tree blocks and 8x8 minimum coding blocks.
  EXPECT_EQ(frame.seq_param.log2_min_luma_coding_block_size_minus3, 0);
  EXPECT_EQ(frame.seq_param.log2_diff_max_min_luma_coding_block_size, 1);
  EXPECT_EQ(frame.seq_param.log2_min_transform_block_size_minus2, 0);
  EXPECT_EQ(frame.seq_param.log2_diff_max_min_transform_block_size, 0);
  EXPECT_EQ(frame.seq_param.max_transform_hierarchy_depth_inter, 0);
  EXPECT_EQ(frame.seq_param.max_transform_hierarchy_depth_intra, 0);
  // 80 x 45 coding tree blocks of 16x16.
  EXPECT_EQ(frame.slice_param.num_ctu_in_slice, 3600u);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, EncodeIdrAndInterFrames) {
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  // The first frame is an IDR picture even if no keyframe is requested.
  const Frame idr_frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_TRUE(idr_frame.is_idr);

  const VAEncSequenceParameterBufferHEVC& seq_param = idr_frame.seq_param;
  EXPECT_EQ(seq_param.general_profile_idc,
            H265ProfileTierLevel::kProfileIdcMain);
  EXPECT_EQ(seq_param.general_level_idc, kLevelIdc3_1);
  EXPECT_EQ(seq_param.general_tier_flag, 0);
  EXPECT_EQ(seq_param.intra_period, kIDRPeriod);
  EXPECT_EQ(seq_param.intra_idr_period, kIDRPeriod);
  EXPECT_EQ(seq_param.ip_period, 1u);
  EXPECT_EQ(seq_param.bits_per_second, kDefaultBitrate);
  EXPECT_EQ(seq_param.pic_width_in_luma_samples, kDefaultVisibleSize.width());
  EXPECT_EQ(seq_param.pic_height_in_luma_samples, kDefaultVisibleSize.height());
  // The default coding tools are used because the driver doesn't report the
  // ones it supports.
  EXPECT_TRUE(seq_param.seq_fields.bits.amp_enabled_flag);
  EXPECT_FALSE(seq_param.seq_fields.bits.sample_adaptive_offset_enabled_flag);
  EXPECT_FALSE(seq_param.seq_fields.bits.sps_temporal_mvp_enabled_flag);
  EXPECT_FALSE(seq_param.seq_fields.bits.strong_intra_smoothing_enabled_flag);

  EXPECT_EQ(idr_frame.rate_control_param.bits_per_second, kDefaultBitrate);
  EXPECT_EQ(idr_frame.rate_control_param.target_percentage, 100u);
  EXPECT_EQ(idr_frame.rate_control_param.window_size, kCPBWindowSizeMs);
  EXPECT_EQ(idr_frame.rate_control_param.initial_qp, 26u);
  EXPECT_EQ(idr_frame.rate_control_param.min_qp, 1u);
  EXPECT_EQ(idr_frame.rate_control_param.max_qp, 51u);
  EXPECT_TRUE(idr_frame.rate_control_param.rc_flags.bits.disable_frame_skip);
  EXPECT_EQ(idr_frame.framerate_param.framerate, kDefaultFramerate);
  EXPECT_EQ(idr_frame.hrd_param.buffer_size,
            kDefaultBitrate / 1000 * kCPBWindowSizeMs);
  EXPECT_EQ(idr_frame.hrd_param.initial_buffer_fullness,
            idr_frame.hrd_param.buffer_size / 2);

  {
    const VAEncPictureParameterBufferHEVC& pic_param = idr_frame.pic_param;
    EXPECT_NE(pic_param.decoded_curr_pic.picture_id, VA_INVALID_ID);
    EXPECT_EQ(pic_param.decoded_curr_pic.pic_order_cnt, 0);
    EXPECT_EQ(pic_param.decoded_curr_pic.flags, 0u);
    for (const VAPictureHEVC& reference_frame : pic_param.reference_frames) {
      EXPECT_TRUE(IsInvalidVAPicture(reference_frame));
    }
    EXPECT_EQ(pic_param.collocated_ref_pic_index, 0xff);
    EXPECT_EQ(pic_param.pic_init_qp, 26);
    EXPECT_EQ(pic_param.nal_unit_type, H265NALU::IDR_W_RADL);
    EXPECT_EQ(pic_param.pic_fields.bits.coding_type, 1u);
    EXPECT_TRUE(pic_param.pic_fields.bits.reference_pic_flag);
    EXPECT_TRUE(pic_param.pic_fields.bits.cu_qp_delta_enabled_flag);

    const VAEncSliceParameterBufferHEVC& slice_param = idr_frame.slice_param;
    EXPECT_EQ(slice_param.slice_segment_address, 0u);
    // 40 x 23 coding tree blocks of 32x32.
    EXPECT_EQ(slice_param.num_ctu_in_slice, 920u);
    EXPECT_EQ(slice_param.slice_type, H265SliceHeader::kSliceTypeI);
    for (const VAPictureHEVC& va_pic : slice_param.ref_pic_list0) {
      EXPECT_TRUE(IsInvalidVAPicture(va_pic));
    }
    for (const VAPictureHEVC& va_pic : slice_param.ref_pic_list1) {
      EXPECT_TRUE(IsInvalidVAPicture(va_pic));
    }
    EXPECT_EQ(slice_param.max_num_merge_cand, 5);
    EXPECT_EQ(slice_param.slice_qp_delta, 0);
    EXPECT_TRUE(slice_param.slice_fields.bits.last_slice_of_pic_flag);
    EXPECT_FALSE(slice_param.slice_fields.bits.slice_temporal_mvp_enabled_flag);
  }

  // The following frames are P pictures that reference the previous picture.
  VAPictureHEVC prev_pic = idr_frame.pic_param.decoded_curr_pic;
  for (int i = 1; i <= 3; ++i) {
    SCOPED_TRACE(i);
    const Frame frame = EncodeFrame(/*force_keyframe=*/false);
    EXPECT_FALSE(frame.is_idr);

    const VAEncPictureParameterBufferHEVC& pic_param = frame.pic_param;
    EXPECT_NE(pic_param.decoded_curr_pic.picture_id, VA_INVALID_ID);
    EXPECT_NE(pic_param.decoded_curr_pic.picture_id, prev_pic.picture_id);
    EXPECT_EQ(pic_param.decoded_curr_pic.pic_order_cnt, i);
    EXPECT_EQ(pic_param.reference_frames[0].picture_id, prev_pic.picture_id);
    EXPECT_EQ(pic_param.reference_frames[0].pic_order_cnt,
              prev_pic.pic_order_cnt);
    EXPECT_EQ(pic_param.reference_frames[0].flags,
              static_cast<uint32_t>(VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE));
    for (const VAPictureHEVC& reference_frame :
         base::span(pic_param.reference_frames).subspan(1u)) {
      EXPECT_TRUE(IsInvalidVAPicture(reference_frame));
    }
    // The temporal motion vector prediction is disabled.
    EXPECT_EQ(pic_param.collocated_ref_pic_index, 0xff);
    EXPECT_EQ(pic_param.nal_unit_type, H265NALU::TRAIL_R);
    EXPECT_EQ(pic_param.pic_fields.bits.coding_type, 2u);
    EXPECT_TRUE(pic_param.pic_fields.bits.reference_pic_flag);

    const VAEncSliceParameterBufferHEVC& slice_param = frame.slice_param;
    EXPECT_EQ(slice_param.slice_type, H265SliceHeader::kSliceTypeP);
    EXPECT_EQ(slice_param.num_ref_idx_l0_active_minus1, 0);
    EXPECT_TRUE(AreSameVAPictures(slice_param.ref_pic_list0[0],
                                  pic_param.reference_frames[0]));
    for (const VAPictureHEVC& va_pic :
         base::span(slice_param.ref_pic_list0).subspan(1u)) {
      EXPECT_TRUE(IsInvalidVAPicture(va_pic));
    }
    for (const VAPictureHEVC& va_pic : slice_param.ref_pic_list1) {
      EXPECT_TRUE(IsInvalidVAPicture(va_pic));
    }
    EXPECT_TRUE(slice_param.slice_fields.bits.last_slice_of_pic_flag);

    ASSERT_TRUE(frame.slice_header);
    EXPECT_EQ(frame.slice_header->slice_type, H265SliceHeader::kSliceTypeP);

    prev_pic = pic_param.decoded_curr_pic;
  }
}

TEST_F(H265VaapiVideoEncoderDelegateTest, KeyframeRequestProducesIdr) {
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  EXPECT_TRUE(EncodeFrame(/*force_keyframe=*/true).is_idr);
  EXPECT_FALSE(EncodeFrame(/*force_keyframe=*/false).is_idr);
  EXPECT_FALSE(EncodeFrame(/*force_keyframe=*/false).is_idr);

  // The picture order count is reset by the requested IDR picture.
  const Frame idr_frame = EncodeFrame(/*force_keyframe=*/true);
  EXPECT_TRUE(idr_frame.is_idr);
  EXPECT_EQ(idr_frame.pic_param.decoded_curr_pic.pic_order_cnt, 0);

  // The next picture references the IDR picture.
  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_FALSE(frame.is_idr);
  EXPECT_EQ(frame.pic_param.decoded_curr_pic.pic_order_cnt, 1);
  EXPECT_TRUE(AreSameVAPictures(frame.slice_param.ref_pic_list0[0],
                                frame.pic_param.reference_frames[0]));
  EXPECT_EQ(frame.pic_param.reference_frames[0].picture_id,
            idr_frame.pic_param.decoded_curr_pic.picture_id);
  EXPECT_EQ(frame.pic_param.reference_frames[0].pic_order_cnt, 0);

  // Consecutive keyframe requests produce consecutive IDR pictures.
  EXPECT_TRUE(EncodeFrame(/*force_keyframe=*/true).is_idr);
  EXPECT_TRUE(EncodeFrame(/*force_keyframe=*/true).is_idr);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, IdrPeriod) {
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  for (uint32_t i = 0; i < kIDRPeriod; ++i) {
    const Frame frame = EncodeFrame(/*force_keyframe=*/false);
    ASSERT_EQ(frame.is_idr, i == 0) << "frame " << i;
    ASSERT_EQ(frame.pic_param.decoded_curr_pic.pic_order_cnt,
              base::checked_cast<int32_t>(i));
    if (::testing::Test::HasFailure()) {
      return;
    }
  }

  // An IDR picture is produced once per kIDRPeriod frames.
  const Frame idr_frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_TRUE(idr_frame.is_idr);
  EXPECT_EQ(idr_frame.pic_param.decoded_curr_pic.pic_order_cnt, 0);
  EXPECT_FALSE(EncodeFrame(/*force_keyframe=*/false).is_idr);
}

TEST_F(H265VaapiVideoEncoderDelegateTest,
       UpdateRatesKeepsGopIfLevelIsUnchanged) {
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));
  EXPECT_TRUE(EncodeFrame(/*force_keyframe=*/false).is_idr);
  EXPECT_FALSE(EncodeFrame(/*force_keyframe=*/false).is_idr);

  // The new bitrate and framerate are applied without an IDR picture.
  constexpr uint32_t kNewBitrate = 1'000'000u;
  constexpr uint32_t kNewFramerate = 15u;
  EXPECT_TRUE(UpdateRates(kNewBitrate, kNewFramerate));
  EXPECT_EQ(level_idc(), kLevelIdc3_1);

  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_FALSE(frame.is_idr);
  EXPECT_EQ(frame.pic_param.decoded_curr_pic.pic_order_cnt, 2);
  EXPECT_EQ(frame.seq_param.general_level_idc, kLevelIdc3_1);
  EXPECT_EQ(frame.seq_param.bits_per_second, kNewBitrate);
  EXPECT_EQ(frame.rate_control_param.bits_per_second, kNewBitrate);
  EXPECT_EQ(frame.framerate_param.framerate, kNewFramerate);
  EXPECT_EQ(frame.hrd_param.buffer_size, kNewBitrate / 1000 * kCPBWindowSizeMs);

  // Setting the same rates again doesn't change anything.
  EXPECT_TRUE(UpdateRates(kNewBitrate, kNewFramerate));
  EXPECT_FALSE(EncodeFrame(/*force_keyframe=*/false).is_idr);
}

TEST_F(H265VaapiVideoEncoderDelegateTest,
       UpdateRatesProducesIdrIfLevelIsRaised) {
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));
  EXPECT_TRUE(EncodeFrame(/*force_keyframe=*/false).is_idr);
  EXPECT_FALSE(EncodeFrame(/*force_keyframe=*/false).is_idr);

  // The new bitrate exceeds the limit of the level 3.1, so the level in the
  // parameter sets has to be raised. They are sent with an IDR picture.
  constexpr uint32_t kHighBitrate = 11'000'000u;
  EXPECT_TRUE(UpdateRates(kHighBitrate, kDefaultFramerate));
  EXPECT_EQ(level_idc(), kLevelIdc4);

  const Frame idr_frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_TRUE(idr_frame.is_idr);
  EXPECT_EQ(idr_frame.pic_param.decoded_curr_pic.pic_order_cnt, 0);
  EXPECT_EQ(idr_frame.seq_param.general_level_idc, kLevelIdc4);
  EXPECT_EQ(idr_frame.seq_param.bits_per_second, kHighBitrate);
  const H265SPS* sps = parsed_sps();
  ASSERT_TRUE(sps);
  EXPECT_EQ(sps->profile_tier_level.general_level_idc, kLevelIdc4);
  EXPECT_FALSE(EncodeFrame(/*force_keyframe=*/false).is_idr);

  // The level is not lowered again, so no IDR picture is produced.
  EXPECT_TRUE(UpdateRates(kDefaultBitrate, kDefaultFramerate));
  EXPECT_EQ(level_idc(), kLevelIdc4);
  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_FALSE(frame.is_idr);
  EXPECT_EQ(frame.seq_param.general_level_idc, kLevelIdc4);
  EXPECT_EQ(frame.seq_param.bits_per_second, kDefaultBitrate);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, UpdateRatesFailsForInvalidRates) {
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));
  EXPECT_TRUE(EncodeFrame(/*force_keyframe=*/false).is_idr);

  EXPECT_FALSE(UpdateRates(0u, kDefaultFramerate));
  EXPECT_FALSE(UpdateRates(kDefaultBitrate, 0u));
  // No level allows such a bitrate.
  EXPECT_FALSE(UpdateRates(1'000'000'000u, kDefaultFramerate));
  // No level allows such a framerate at this resolution.
  EXPECT_FALSE(UpdateRates(kDefaultBitrate, 100'000u));
  // The bitrate mode can't be changed.
  VideoBitrateAllocation variable_bitrate_allocation(Bitrate::Mode::kVariable);
  EXPECT_TRUE(variable_bitrate_allocation.SetBitrate(0, 0, kDefaultBitrate));
  EXPECT_FALSE(
      encoder_->UpdateRates(variable_bitrate_allocation, kDefaultFramerate));

  // The failed updates don't change the encoding parameters.
  EXPECT_EQ(level_idc(), kLevelIdc3_1);
  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_FALSE(frame.is_idr);
  EXPECT_EQ(frame.seq_param.bits_per_second, kDefaultBitrate);
  EXPECT_EQ(frame.rate_control_param.bits_per_second, kDefaultBitrate);
  EXPECT_EQ(frame.framerate_param.framerate, kDefaultFramerate);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, ScreenContentUsesHigherMinQp) {
  auto config = DefaultVEAConfig();
  config.content_type = VideoEncodeAccelerator::Config::ContentType::kDisplay;
  ASSERT_TRUE(InitializeEncoder(config));

  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_EQ(frame.rate_control_param.min_qp, 10u);
  EXPECT_EQ(frame.rate_control_param.max_qp, 51u);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, GpbSlicesIfPSlicesAreUnsupported) {
  // The driver only supports inter slices with two non-empty reference picture
  // lists.
  mock_vaapi_wrapper_->SetAttribute(VAConfigAttribPredictionDirection,
                                    VA_PREDICTION_DIRECTION_PREVIOUS |
                                        VA_PREDICTION_DIRECTION_FUTURE |
                                        VA_PREDICTION_DIRECTION_BI_NOT_EMPTY);
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  const Frame idr_frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_TRUE(idr_frame.is_idr);
  EXPECT_EQ(idr_frame.slice_param.slice_type, H265SliceHeader::kSliceTypeI);
  for (const VAPictureHEVC& va_pic : idr_frame.slice_param.ref_pic_list1) {
    EXPECT_TRUE(IsInvalidVAPicture(va_pic));
  }

  for (int i = 1; i <= 3; ++i) {
    SCOPED_TRACE(i);
    const Frame frame = EncodeFrame(/*force_keyframe=*/false);
    EXPECT_FALSE(frame.is_idr);

    // The inter picture is a P picture encoded with a B slice whose reference
    // picture lists are identical.
    const VAEncPictureParameterBufferHEVC& pic_param = frame.pic_param;
    EXPECT_EQ(pic_param.pic_fields.bits.coding_type, 2u);
    EXPECT_EQ(pic_param.nal_unit_type, H265NALU::TRAIL_R);
    EXPECT_FALSE(IsInvalidVAPicture(pic_param.reference_frames[0]));
    for (const VAPictureHEVC& reference_frame :
         base::span(pic_param.reference_frames).subspan(1u)) {
      EXPECT_TRUE(IsInvalidVAPicture(reference_frame));
    }

    const VAEncSliceParameterBufferHEVC& slice_param = frame.slice_param;
    EXPECT_EQ(slice_param.slice_type, H265SliceHeader::kSliceTypeB);
    EXPECT_EQ(slice_param.num_ref_idx_l0_active_minus1, 0);
    EXPECT_EQ(slice_param.num_ref_idx_l1_active_minus1, 0);
    EXPECT_TRUE(AreSameVAPictures(slice_param.ref_pic_list0[0],
                                  pic_param.reference_frames[0]));
    EXPECT_TRUE(AreSameVAPictures(slice_param.ref_pic_list1[0],
                                  pic_param.reference_frames[0]));
    for (const VAPictureHEVC& va_pic :
         base::span(slice_param.ref_pic_list1).subspan(1u)) {
      EXPECT_TRUE(IsInvalidVAPicture(va_pic));
    }

    // The slice segment header is the one of a B slice, which is verified
    // against the slice parameters by EncodeFrame().
    ASSERT_TRUE(frame.slice_header);
    EXPECT_EQ(frame.slice_header->slice_type, H265SliceHeader::kSliceTypeB);
    EXPECT_FALSE(frame.slice_header->mvd_l1_zero_flag);
  }
}

TEST_F(H265VaapiVideoEncoderDelegateTest, PSlicesIfSupported) {
  // The driver supports P slices, so no GPB slices are needed.
  mock_vaapi_wrapper_->SetAttribute(
      VAConfigAttribPredictionDirection,
      VA_PREDICTION_DIRECTION_PREVIOUS | VA_PREDICTION_DIRECTION_FUTURE);
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  EXPECT_TRUE(EncodeFrame(/*force_keyframe=*/false).is_idr);
  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_EQ(frame.slice_param.slice_type, H265SliceHeader::kSliceTypeP);
  for (const VAPictureHEVC& va_pic : frame.slice_param.ref_pic_list1) {
    EXPECT_TRUE(IsInvalidVAPicture(va_pic));
  }
}

TEST_F(H265VaapiVideoEncoderDelegateTest, CodingToolsReportedByDriver) {
  mock_vaapi_wrapper_->SetAttribute(
      VAConfigAttribEncHEVCBlockSizes,
      MakeBlockSizesAttribute(/*log2_ctb_size=*/6, /*log2_min_cb_size=*/3,
                              /*log2_max_tb_size=*/5, /*log2_min_tb_size=*/2,
                              /*max_transform_hierarchy_depth=*/2));
  mock_vaapi_wrapper_->SetAttribute(VAConfigAttribEncHEVCFeatures,
                                    MakeFeaturesAttribute());
  mock_vaapi_wrapper_->SetAttribute(VAConfigAttribPredictionDirection,
                                    VA_PREDICTION_DIRECTION_PREVIOUS |
                                        VA_PREDICTION_DIRECTION_FUTURE |
                                        VA_PREDICTION_DIRECTION_BI_NOT_EMPTY);
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));
  EXPECT_EQ(encoder_->GetCodedSize(), kDefaultVisibleSize);

  const Frame idr_frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_TRUE(idr_frame.is_idr);
  const auto& seq_fields = idr_frame.seq_param.seq_fields.bits;
  EXPECT_TRUE(seq_fields.amp_enabled_flag);
  EXPECT_TRUE(seq_fields.sample_adaptive_offset_enabled_flag);
  EXPECT_TRUE(seq_fields.sps_temporal_mvp_enabled_flag);
  EXPECT_FALSE(seq_fields.strong_intra_smoothing_enabled_flag);
  EXPECT_TRUE(idr_frame.pic_param.pic_fields.bits.transform_skip_enabled_flag);
  EXPECT_TRUE(idr_frame.pic_param.pic_fields.bits.cu_qp_delta_enabled_flag);
  // The QP can change on every minimum coding block.
  EXPECT_EQ(idr_frame.pic_param.diff_cu_qp_delta_depth, 3);
  // 20 x 12 coding tree blocks of 64x64.
  EXPECT_EQ(idr_frame.slice_param.num_ctu_in_slice, 240u);
  // There is no collocated picture in an IDR picture.
  EXPECT_EQ(idr_frame.pic_param.collocated_ref_pic_index, 0xff);
  const auto& idr_slice_fields = idr_frame.slice_param.slice_fields.bits;
  EXPECT_FALSE(idr_slice_fields.slice_temporal_mvp_enabled_flag);
  EXPECT_TRUE(idr_slice_fields.slice_sao_luma_flag);
  EXPECT_TRUE(idr_slice_fields.slice_sao_chroma_flag);

  for (int i = 1; i <= 3; ++i) {
    SCOPED_TRACE(i);
    const Frame frame = EncodeFrame(/*force_keyframe=*/false);
    EXPECT_FALSE(frame.is_idr);
    // The collocated picture is the only reference picture.
    EXPECT_EQ(frame.pic_param.collocated_ref_pic_index, 0);
    EXPECT_EQ(frame.slice_param.slice_type, H265SliceHeader::kSliceTypeB);
    const auto& slice_fields = frame.slice_param.slice_fields.bits;
    EXPECT_TRUE(slice_fields.slice_temporal_mvp_enabled_flag);
    EXPECT_TRUE(slice_fields.collocated_from_l0_flag);
    EXPECT_TRUE(slice_fields.slice_sao_luma_flag);
    EXPECT_TRUE(slice_fields.slice_sao_chroma_flag);

    // The slice segment header carries the syntax elements of the enabled
    // coding tools, which is verified by EncodeFrame().
    ASSERT_TRUE(frame.slice_header);
    EXPECT_TRUE(frame.slice_header->slice_temporal_mvp_enabled_flag);
    EXPECT_TRUE(frame.slice_header->slice_sao_luma_flag);
    EXPECT_TRUE(frame.slice_header->slice_sao_chroma_flag);
  }
}

TEST_F(H265VaapiVideoEncoderDelegateTest, CodingToolsUnsupportedByDriver) {
  // The driver reports that it doesn't support any optional coding tool.
  mock_vaapi_wrapper_->SetAttribute(VAConfigAttribEncHEVCFeatures, 0u);
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  const Frame idr_frame = EncodeFrame(/*force_keyframe=*/false);
  const auto& seq_fields = idr_frame.seq_param.seq_fields.bits;
  EXPECT_FALSE(seq_fields.amp_enabled_flag);
  EXPECT_FALSE(seq_fields.sample_adaptive_offset_enabled_flag);
  EXPECT_FALSE(seq_fields.sps_temporal_mvp_enabled_flag);
  EXPECT_FALSE(seq_fields.strong_intra_smoothing_enabled_flag);
  EXPECT_FALSE(idr_frame.pic_param.pic_fields.bits.transform_skip_enabled_flag);
  EXPECT_FALSE(idr_frame.pic_param.pic_fields.bits.cu_qp_delta_enabled_flag);
  EXPECT_EQ(idr_frame.pic_param.diff_cu_qp_delta_depth, 0);

  EXPECT_FALSE(EncodeFrame(/*force_keyframe=*/false).is_idr);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, NoPackedHeaders) {
  SetSupportedPackedHeaders(VA_ENC_PACKED_HEADER_NONE);
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  // EncodeFrame() verifies that only the parameter buffers are submitted.
  const Frame idr_frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_TRUE(idr_frame.is_idr);
  EXPECT_EQ(idr_frame.buffers.size(), 6u);
  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_FALSE(frame.is_idr);
  EXPECT_EQ(frame.buffers.size(), 6u);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, PackedParameterSetsOnly) {
  SetSupportedPackedHeaders(VA_ENC_PACKED_HEADER_SEQUENCE);
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  // The parameter sets are submitted with the IDR picture only.
  const Frame idr_frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_TRUE(idr_frame.is_idr);
  EXPECT_EQ(idr_frame.buffers.size(), 8u);
  EXPECT_FALSE(idr_frame.slice_header);
  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_FALSE(frame.is_idr);
  EXPECT_EQ(frame.buffers.size(), 6u);
  EXPECT_FALSE(frame.slice_header);
}

TEST_F(H265VaapiVideoEncoderDelegateTest,
       PackedSliceHeaderRequiresPackedParameterSets) {
  // The slice segment header refers to the parameter sets generated by the
  // delegate, so it is not submitted if the driver generates them.
  SetSupportedPackedHeaders(VA_ENC_PACKED_HEADER_SLICE);
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  const Frame idr_frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_TRUE(idr_frame.is_idr);
  EXPECT_EQ(idr_frame.buffers.size(), 6u);
  const Frame frame = EncodeFrame(/*force_keyframe=*/false);
  EXPECT_FALSE(frame.is_idr);
  EXPECT_EQ(frame.buffers.size(), 6u);
}

TEST_F(H265VaapiVideoEncoderDelegateTest, PackedHeadersOnEveryIdr) {
  ASSERT_TRUE(InitializeEncoder(DefaultVEAConfig()));

  // The parameter sets and the slice segment header are submitted with an IDR
  // picture, and only the slice segment header with the other pictures.
  EXPECT_EQ(EncodeFrame(/*force_keyframe=*/false).buffers.size(), 10u);
  EXPECT_EQ(EncodeFrame(/*force_keyframe=*/false).buffers.size(), 8u);
  EXPECT_EQ(EncodeFrame(/*force_keyframe=*/true).buffers.size(), 10u);
  EXPECT_EQ(EncodeFrame(/*force_keyframe=*/false).buffers.size(), 8u);
}

}  // namespace media
