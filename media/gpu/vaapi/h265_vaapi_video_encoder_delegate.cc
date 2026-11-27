// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "media/gpu/vaapi/h265_vaapi_video_encoder_delegate.h"

#include <algorithm>
#include <array>
#include <optional>
#include <utility>

#include "base/bits.h"
#include "base/compiler_specific.h"
#include "base/numerics/checked_math.h"
#include "base/numerics/safe_conversions.h"
#include "media/base/bitrate.h"
#include "media/base/video_codecs.h"
#include "media/gpu/gpu_video_encode_accelerator_helpers.h"
#include "media/gpu/h265_builder.h"
#include "media/gpu/h265_level_utils.h"
#include "media/gpu/macros.h"
#include "media/gpu/vaapi/vaapi_common.h"
#include "media/gpu/vaapi/vaapi_wrapper.h"
#include "media/video/video_encode_accelerator.h"

namespace media {
namespace {

// An IDR every 2048 frames, no I frames and no B frames. The picture order
// count is reset by every IDR, so it is always less than this value.
constexpr uint32_t kIDRPeriod = 2048;
// How often do we need to have either an I or a P frame in the stream.
// A period of 1 implies no B frames.
constexpr uint32_t kIPPeriod = 1;

// log2_max_pic_order_cnt_lsb_minus4 of the SPS. The picture order count LSB
// has to be able to represent all the picture order counts of an IDR period.
constexpr int kLog2MaxPicOrderCntLsbMinus4 = 8;
static_assert(kIDRPeriod <= (1u << (kLog2MaxPicOrderCntLsbMinus4 + 4)),
              "The picture order count LSB must not wrap in an IDR period");

// The decoded picture buffer has to hold the current picture and the only
// reference picture.
constexpr int kMaxDecPicBufferingMinus1 = 1;

// The qp range is 0-51 in H265. Select 26 because of the center value. The
// minimum QP is 1 for camera and 10 for screen sharing to mitigate the bitrate
// overshoot due to a scene change.
constexpr uint8_t kDefaultQP = 26;
constexpr uint8_t kMinQP = 1;
constexpr uint8_t kScreenMinQP = 10;
constexpr uint8_t kMaxQP = 51;

// Subjectively chosen bitrate window size for rate control, in ms.
constexpr uint32_t kCPBWindowSizeMs = 1500;

// 4:2:0
constexpr int kChromaFormatIDC = 1;

// MaxNumMergeCand (spec section 7.4.7.1).
constexpr uint8_t kMaxNumMergeCand = 5;

// The coding block structure used if the driver doesn't report the block
// sizes it supports: 32x32 coding tree blocks, 16x16 minimum coding blocks,
// 4x4 to 32x32 transform blocks and the full transform hierarchy.
constexpr uint32_t kDefaultLog2CtbSize = 5;
constexpr uint32_t kDefaultLog2MinCbSize = 4;
constexpr uint32_t kDefaultLog2MinTbSize = 2;
constexpr uint32_t kDefaultLog2MaxTbSize = 5;
constexpr uint32_t kDefaultMaxTransformHierarchyDepth = 3;

// The limits of the coding block structure (spec section 7.4.3.2.1).
constexpr uint32_t kMinLog2CtbSize = 4;
constexpr uint32_t kMaxLog2CtbSize = 6;
constexpr uint32_t kMinLog2CbSize = 3;
constexpr uint32_t kMinLog2TbSize = 2;
constexpr uint32_t kMaxLog2TbSize = 5;

template <typename VAEncMiscParam>
VAEncMiscParam& AllocateMiscParameterBuffer(
    std::vector<uint8_t>& misc_buffer,
    VAEncMiscParameterType misc_param_type) {
  constexpr size_t buffer_size =
      sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParam);
  misc_buffer.resize(buffer_size);
  // SAFETY: |misc_buffer| is resized above to hold a VAEncMiscParameterBuffer
  // that is followed by a VAEncMiscParam.
  auto* va_buffer = UNSAFE_BUFFERS(
      reinterpret_cast<VAEncMiscParameterBuffer*>(misc_buffer.data()));
  va_buffer->type = misc_param_type;
  return *reinterpret_cast<VAEncMiscParam*>(va_buffer->data);
}

void CreateVAEncRateControlParams(
    uint32_t bps,
    uint32_t window_size,
    uint32_t initial_qp,
    uint32_t min_qp,
    uint32_t max_qp,
    uint32_t framerate,
    uint32_t buffer_size,
    std::array<std::vector<uint8_t>, 3>& buffers) {
  auto& rate_control_param =
      AllocateMiscParameterBuffer<VAEncMiscParameterRateControl>(
          buffers[0], VAEncMiscParameterTypeRateControl);
  rate_control_param.bits_per_second = bps;
  // The target bitrate is the maximum bitrate in constant bitrate mode.
  rate_control_param.target_percentage = 100u;
  rate_control_param.window_size = window_size;
  rate_control_param.initial_qp = initial_qp;
  rate_control_param.min_qp = min_qp;
  rate_control_param.max_qp = max_qp;
  rate_control_param.rc_flags.bits.disable_frame_skip = true;

  auto& framerate_param =
      AllocateMiscParameterBuffer<VAEncMiscParameterFrameRate>(
          buffers[1], VAEncMiscParameterTypeFrameRate);
  framerate_param.framerate = framerate;

  auto& hrd_param = AllocateMiscParameterBuffer<VAEncMiscParameterHRD>(
      buffers[2], VAEncMiscParameterTypeHRD);
  hrd_param.buffer_size = buffer_size;
  hrd_param.initial_buffer_fullness = buffer_size / 2;
}

void InitVAPictureHEVC(VAPictureHEVC& va_pic) {
  va_pic = {};
  va_pic.picture_id = VA_INVALID_ID;
  va_pic.flags = VA_PICTURE_HEVC_INVALID;
}

scoped_refptr<H265Picture> GetH265Picture(
    const VaapiVideoEncoderDelegate::EncodeJob& job) {
  return base::WrapRefCounted(static_cast<H265Picture*>(job.picture().get()));
}

}  // namespace

H265VaapiVideoEncoderDelegate::EncodeParams::EncodeParams()
    : framerate(0),
      cpb_window_size_ms(kCPBWindowSizeMs),
      cpb_size_bits(0),
      initial_qp(kDefaultQP),
      min_qp(kMinQP),
      max_qp(kMaxQP) {}

H265VaapiVideoEncoderDelegate::H265VaapiVideoEncoderDelegate(
    scoped_refptr<VaapiWrapper> vaapi_wrapper,
    base::RepeatingClosure error_cb)
    : VaapiVideoEncoderDelegate(std::move(vaapi_wrapper), error_cb) {}

H265VaapiVideoEncoderDelegate::~H265VaapiVideoEncoderDelegate() = default;

bool H265VaapiVideoEncoderDelegate::Initialize(
    const VideoEncodeAccelerator::Config& config,
    const VaapiVideoEncoderDelegate::Config& ave_config) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);

  if (config.output_profile != HEVCPROFILE_MAIN) {
    DVLOGF(1) << "Unsupported profile "
              << GetProfileName(config.output_profile);
    return false;
  }

  if (config.input_visible_size.IsEmpty()) {
    DVLOGF(1) << "Input visible size could not be empty";
    return false;
  }

  if (config.HasSpatialLayer() || config.HasTemporalLayer()) {
    DVLOGF(1) << "Spatial and temporal layer encoding are not supported";
    return false;
  }

  if (config.bitrate.mode() != Bitrate::Mode::kConstant) {
    DVLOGF(1) << "Only constant bitrate encoding is supported";
    return false;
  }

  visible_size_ = config.input_visible_size;
  // For 4:2:0, the pixel sizes have to be even.
  if ((visible_size_.width() % 2 != 0) || (visible_size_.height() % 2 != 0)) {
    DVLOGF(1) << "The pixel sizes are not even: " << visible_size_.ToString();
    return false;
  }

  // Query the coding block structure supported by the driver.
  uint32_t log2_ctb_size = kDefaultLog2CtbSize;
  uint32_t log2_min_cb_size = kDefaultLog2MinCbSize;
  uint32_t log2_min_tb_size = kDefaultLog2MinTbSize;
  uint32_t log2_max_tb_size = kDefaultLog2MaxTbSize;
  uint32_t max_transform_hierarchy_depth_inter =
      kDefaultMaxTransformHierarchyDepth;
  uint32_t max_transform_hierarchy_depth_intra =
      kDefaultMaxTransformHierarchyDepth;
  uint32_t attrib_value = 0;
  if (vaapi_wrapper_->GetEncConfigAttribute(config.output_profile,
                                            VAConfigAttribEncHEVCBlockSizes,
                                            &attrib_value)) {
    VAConfigAttribValEncHEVCBlockSizes block_sizes;
    block_sizes.value = attrib_value;
    log2_ctb_size = block_sizes.bits.log2_max_coding_tree_block_size_minus3 + 3;
    log2_min_cb_size =
        block_sizes.bits.log2_min_luma_coding_block_size_minus3 + 3;
    log2_min_tb_size =
        block_sizes.bits.log2_min_luma_transform_block_size_minus2 + 2;
    log2_max_tb_size =
        block_sizes.bits.log2_max_luma_transform_block_size_minus2 + 2;
    max_transform_hierarchy_depth_inter =
        block_sizes.bits.max_max_transform_hierarchy_depth_inter;
    max_transform_hierarchy_depth_intra =
        block_sizes.bits.max_max_transform_hierarchy_depth_intra;
  } else {
    DVLOGF(2) << "The driver doesn't report the supported block sizes, using "
              << "the default ones";
  }
  // Make sure that the block sizes fulfill the constraints of the spec even
  // if a driver reports something unexpected.
  log2_ctb_size = std::clamp(log2_ctb_size, kMinLog2CtbSize, kMaxLog2CtbSize);
  log2_min_cb_size =
      std::clamp(log2_min_cb_size, kMinLog2CbSize, log2_ctb_size);
  log2_min_tb_size =
      std::clamp(log2_min_tb_size, kMinLog2TbSize, log2_min_cb_size - 1);
  log2_max_tb_size = std::clamp(log2_max_tb_size, log2_min_tb_size,
                                std::min(log2_ctb_size, kMaxLog2TbSize));
  max_transform_hierarchy_depth_inter = std::min(
      max_transform_hierarchy_depth_inter, log2_ctb_size - log2_min_tb_size);
  max_transform_hierarchy_depth_intra = std::min(
      max_transform_hierarchy_depth_intra, log2_ctb_size - log2_min_tb_size);

  // Query the coding tools supported or required by the driver.
  bool amp_enabled = true;
  bool sao_enabled = false;
  bool temporal_mvp_enabled = false;
  bool strong_intra_smoothing_enabled = false;
  bool transform_skip_enabled = false;
  // The driver rate controller needs to be able to change the QP inside a
  // picture.
  bool cu_qp_delta_enabled = true;
  uint32_t diff_cu_qp_delta_depth = 0;
  if (vaapi_wrapper_->GetEncConfigAttribute(config.output_profile,
                                            VAConfigAttribEncHEVCFeatures,
                                            &attrib_value)) {
    VAConfigAttribValEncHEVCFeatures features;
    features.value = attrib_value;
    amp_enabled = features.bits.amp != VA_FEATURE_NOT_SUPPORTED;
    sao_enabled = features.bits.sao != VA_FEATURE_NOT_SUPPORTED;
    temporal_mvp_enabled =
        features.bits.temporal_mvp != VA_FEATURE_NOT_SUPPORTED;
    strong_intra_smoothing_enabled =
        features.bits.strong_intra_smoothing == VA_FEATURE_REQUIRED;
    transform_skip_enabled =
        features.bits.transform_skip != VA_FEATURE_NOT_SUPPORTED;
    cu_qp_delta_enabled = features.bits.cu_qp_delta != VA_FEATURE_NOT_SUPPORTED;
    if (cu_qp_delta_enabled) {
      // Allow the QP to change on every minimum coding block.
      diff_cu_qp_delta_depth = log2_ctb_size - log2_min_cb_size;
    }
  } else {
    DVLOGF(2) << "The driver doesn't report the supported coding tools, using "
              << "the default ones";
  }

  // Query the prediction directions supported by the driver.
  use_gpb_ = false;
  if (vaapi_wrapper_->GetEncConfigAttribute(config.output_profile,
                                            VAConfigAttribPredictionDirection,
                                            &attrib_value)) {
    use_gpb_ = (attrib_value & VA_PREDICTION_DIRECTION_BI_NOT_EMPTY) != 0;
  }
  DVLOGF_IF(2, use_gpb_) << "P slices are not supported by the driver, "
                         << "encoding GPB slices instead";

  bool packed_sequence_header = false;
  bool packed_picture_header = false;
  bool packed_slice_header = false;
  if (!vaapi_wrapper_->GetSupportedPackedHeaders(
          config.output_profile, packed_sequence_header, packed_picture_header,
          packed_slice_header)) {
    DVLOGF(1) << "Failed getting supported packed headers";
    return false;
  }
  // This has to match the packed headers requested on creating the VA config,
  // see GetRequiredAttribs() in vaapi_wrapper.cc. The packed slice segment
  // header is only valid along with the parameter sets generated here.
  submit_packed_parameter_sets_ = packed_sequence_header;
  submit_packed_slice_header_ = packed_sequence_header && packed_slice_header;
  DVLOGF_IF(2, !submit_packed_parameter_sets_)
      << "Packed parameter sets are not submitted to a driver";
  DVLOGF_IF(2, !submit_packed_slice_header_)
      << "Packed slice header is not submitted to a driver";

  // The picture sizes have to be a multiple of the minimum coding block size.
  // The padding is cropped by the conformance window.
  const uint32_t min_cb_size = 1u << log2_min_cb_size;
  const uint32_t ctb_size = 1u << log2_ctb_size;
  const uint32_t coded_width = base::bits::AlignUp(
      base::checked_cast<uint32_t>(visible_size_.width()), min_cb_size);
  const uint32_t coded_height = base::bits::AlignUp(
      base::checked_cast<uint32_t>(visible_size_.height()), min_cb_size);
  coded_size_ = gfx::Size(base::checked_cast<int>(coded_width),
                          base::checked_cast<int>(coded_height));
  num_ctus_in_pic_ = (base::bits::AlignUp(coded_width, ctb_size) / ctb_size) *
                     (base::bits::AlignUp(coded_height, ctb_size) / ctb_size);

  // Fill the parameter sets. Only the level, which is set in
  // UpdateParameterSets(), can change afterwards.
  // Video parameter set (spec section 7.3.2.1).
  vps_.vps_video_parameter_set_id = 0;
  vps_.vps_base_layer_internal_flag = true;
  vps_.vps_base_layer_available_flag = true;
  vps_.vps_max_layers_minus1 = 0;
  vps_.vps_max_sub_layers_minus1 = 0;
  vps_.vps_temporal_id_nesting_flag = true;
  H265ProfileTierLevel& profile_tier_level = vps_.profile_tier_level;
  profile_tier_level.general_profile_idc =
      H265ProfileTierLevel::kProfileIdcMain;
  // general_profile_compatibility_flag[general_profile_idc] is set.
  profile_tier_level.general_profile_compatibility_flags =
      1u << (31 - H265ProfileTierLevel::kProfileIdcMain);
  profile_tier_level.general_progressive_source_flag = true;
  profile_tier_level.general_interlaced_source_flag = false;
  profile_tier_level.general_non_packed_constraint_flag = true;
  profile_tier_level.general_frame_only_constraint_flag = true;
  vps_.vps_max_dec_pic_buffering_minus1[0] = kMaxDecPicBufferingMinus1;
  vps_.vps_max_num_reorder_pics[0] = 0;
  // The pictures are output in the decoding order without any latency.
  vps_.vps_max_latency_increase_plus1[0] = 1;
  vps_.vps_max_layer_id = 0;
  vps_.vps_num_layer_sets_minus1 = 0;

  // Sequence parameter set (spec section 7.3.2.2).
  sps_.sps_video_parameter_set_id = vps_.vps_video_parameter_set_id;
  sps_.sps_max_sub_layers_minus1 = vps_.vps_max_sub_layers_minus1;
  sps_.sps_temporal_id_nesting_flag = vps_.vps_temporal_id_nesting_flag;
  sps_.profile_tier_level = vps_.profile_tier_level;
  sps_.sps_seq_parameter_set_id = 0;
  sps_.chroma_format_idc = kChromaFormatIDC;
  sps_.separate_colour_plane_flag = false;
  sps_.pic_width_in_luma_samples = coded_size_.width();
  sps_.pic_height_in_luma_samples = coded_size_.height();
  // The conformance window offsets are in units of chroma samples.
  sps_.conf_win_left_offset = 0;
  sps_.conf_win_right_offset =
      (coded_size_.width() - visible_size_.width()) >> 1;
  sps_.conf_win_top_offset = 0;
  sps_.conf_win_bottom_offset =
      (coded_size_.height() - visible_size_.height()) >> 1;
  sps_.bit_depth_luma_minus8 = 0;
  sps_.bit_depth_chroma_minus8 = 0;
  sps_.log2_max_pic_order_cnt_lsb_minus4 = kLog2MaxPicOrderCntLsbMinus4;
  sps_.sps_max_dec_pic_buffering_minus1[0] =
      vps_.vps_max_dec_pic_buffering_minus1[0];
  sps_.sps_max_num_reorder_pics[0] = vps_.vps_max_num_reorder_pics[0];
  sps_.sps_max_latency_increase_plus1[0] =
      vps_.vps_max_latency_increase_plus1[0];
  sps_.log2_min_luma_coding_block_size_minus3 = log2_min_cb_size - 3;
  sps_.log2_diff_max_min_luma_coding_block_size =
      log2_ctb_size - log2_min_cb_size;
  sps_.log2_min_luma_transform_block_size_minus2 = log2_min_tb_size - 2;
  sps_.log2_diff_max_min_luma_transform_block_size =
      log2_max_tb_size - log2_min_tb_size;
  sps_.max_transform_hierarchy_depth_inter =
      max_transform_hierarchy_depth_inter;
  sps_.max_transform_hierarchy_depth_intra =
      max_transform_hierarchy_depth_intra;
  sps_.scaling_list_enabled_flag = false;
  sps_.amp_enabled_flag = amp_enabled;
  sps_.sample_adaptive_offset_enabled_flag = sao_enabled;
  sps_.pcm_enabled_flag = false;
  // The reference picture set is signaled in every slice segment header.
  sps_.num_short_term_ref_pic_sets = 0;
  sps_.long_term_ref_pics_present_flag = false;
  sps_.sps_temporal_mvp_enabled_flag = temporal_mvp_enabled;
  sps_.strong_intra_smoothing_enabled_flag = strong_intra_smoothing_enabled;

  // Picture parameter set (spec section 7.3.2.3).
  pps_.pps_pic_parameter_set_id = 0;
  pps_.pps_seq_parameter_set_id = sps_.sps_seq_parameter_set_id;
  pps_.dependent_slice_segments_enabled_flag = false;
  pps_.output_flag_present_flag = false;
  pps_.num_extra_slice_header_bits = 0;
  pps_.sign_data_hiding_enabled_flag = false;
  pps_.cabac_init_present_flag = false;
  pps_.num_ref_idx_l0_default_active_minus1 = 0;
  pps_.num_ref_idx_l1_default_active_minus1 = 0;
  pps_.init_qp_minus26 = curr_params_.initial_qp - 26;
  pps_.constrained_intra_pred_flag = false;
  pps_.transform_skip_enabled_flag = transform_skip_enabled;
  pps_.cu_qp_delta_enabled_flag = cu_qp_delta_enabled;
  pps_.diff_cu_qp_delta_depth = diff_cu_qp_delta_depth;
  pps_.pps_cb_qp_offset = 0;
  pps_.pps_cr_qp_offset = 0;
  pps_.pps_slice_chroma_qp_offsets_present_flag = false;
  pps_.weighted_pred_flag = false;
  pps_.weighted_bipred_flag = false;
  pps_.transquant_bypass_enabled_flag = false;
  pps_.tiles_enabled_flag = false;
  pps_.entropy_coding_sync_enabled_flag = false;
  pps_.pps_loop_filter_across_slices_enabled_flag = true;
  pps_.deblocking_filter_control_present_flag = false;
  pps_.pps_scaling_list_data_present_flag = false;
  pps_.lists_modification_present_flag = false;
  pps_.log2_parallel_merge_level_minus2 = 0;
  pps_.slice_segment_header_extension_present_flag = false;
  pps_.pps_extension_present_flag = false;

  if (config.content_type ==
      VideoEncodeAccelerator::Config::ContentType::kDisplay) {
    curr_params_.min_qp = kScreenMinQP;
  }

  num_encoded_frames_ = 0;
  ref_pic_.reset();

  // If we don't set the stored BitrateAllocation to the right type, UpdateRates
  // will mistakenly reject the bitrate.
  curr_params_.bitrate_allocation =
      VideoBitrateAllocation(config.bitrate.mode());
  curr_params_.framerate = 0;
  // Force UpdateRates() to select the level and to generate the packed
  // parameter sets.
  level_idc_ = 0;

  return UpdateRates(AllocateBitrateForDefaultEncoding(config),
                     config.framerate);
}

gfx::Size H265VaapiVideoEncoderDelegate::GetCodedSize() const {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  DCHECK(!coded_size_.IsEmpty());

  return coded_size_;
}

size_t H265VaapiVideoEncoderDelegate::GetMaxNumOfRefFrames() const {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);

  // Every inter picture references the previous picture only.
  return 1u;
}

std::vector<gfx::Size> H265VaapiVideoEncoderDelegate::GetSVCLayerResolutions() {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);

  return {visible_size_};
}

BitstreamBufferMetadata H265VaapiVideoEncoderDelegate::GetMetadata(
    const EncodeJob& encode_job,
    size_t payload_size) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  CHECK(!encode_job.IsFrameDropped());
  CHECK_NE(payload_size, 0u);

  // The QP of the frame is unknown because the rate control is done by the
  // driver, so |metadata.qp| is left unset.
  BitstreamBufferMetadata metadata(
      payload_size, encode_job.IsKeyframeRequested(), encode_job.timestamp());
  CHECK(metadata.end_of_picture());
  return metadata;
}

void H265VaapiVideoEncoderDelegate::BitrateControlUpdate(
    const BitstreamBufferMetadata& metadata) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  // Nothing to do, the rate control is done by the driver.
}

VaapiVideoEncoderDelegate::PrepareEncodeJobResult
H265VaapiVideoEncoderDelegate::PrepareEncodeJob(EncodeJob& encode_job) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);

  scoped_refptr<H265Picture> pic = GetH265Picture(encode_job);
  DCHECK(pic);

  if (encode_job.IsKeyframeRequested() || encoding_parameters_changed_) {
    num_encoded_frames_ = 0;
  }

  const bool is_idr = num_encoded_frames_ == 0;
  if (is_idr) {
    // An IDR picture doesn't reference any picture and the pictures following
    // it can't reference any picture preceding it.
    ref_pic_.reset();

    encoding_parameters_changed_ = false;
    encode_job.ProduceKeyframe();
  }
  DCHECK(is_idr || ref_pic_);

  pic->nal_unit_type_ = is_idr ? H265NALU::IDR_W_RADL : H265NALU::TRAIL_R;
  pic->irap_pic_ = is_idr;
  pic->first_picture_ = is_idr;
  pic->no_rasl_output_flag_ = is_idr;
  pic->pic_output_flag_ = true;
  // The picture order count is reset by an IDR picture and is less than
  // kIDRPeriod, so it always equals its LSB.
  pic->pic_order_cnt_val_ = base::checked_cast<int>(num_encoded_frames_);
  pic->pic_order_cnt_msb_ = 0;
  pic->slice_pic_order_cnt_lsb_ = pic->pic_order_cnt_val_;

  DVLOGF(4) << "Starting a new frame, type: " << (is_idr ? "IDR" : "P")
            << (encode_job.IsKeyframeRequested() ? " (keyframe)" : "")
            << " POC: " << pic->pic_order_cnt_val_;

  if (!SubmitFrameParameters(encode_job, *pic, ref_pic_.get())) {
    DVLOGF(1) << "Failed submitting frame parameters";
    return PrepareEncodeJobResult::kFail;
  }

  // Every picture is referenced by the next picture.
  ref_pic_ = std::move(pic);

  num_encoded_frames_++;
  num_encoded_frames_ %= kIDRPeriod;
  return PrepareEncodeJobResult::kSuccess;
}

bool H265VaapiVideoEncoderDelegate::UpdateRates(
    const VideoBitrateAllocation& bitrate_allocation,
    uint32_t framerate) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);

  if (bitrate_allocation.GetMode() !=
      curr_params_.bitrate_allocation.GetMode()) {
    DVLOGF(1) << "Unexpected bitrate mode, requested rate "
              << bitrate_allocation.GetSumBitrate().ToString()
              << ", expected mode to match "
              << curr_params_.bitrate_allocation.GetSumBitrate().ToString();
    return false;
  }

  const uint32_t bitrate = bitrate_allocation.GetSumBps();
  if (bitrate == 0 || framerate == 0) {
    return false;
  }

  if (curr_params_.bitrate_allocation == bitrate_allocation &&
      curr_params_.framerate == framerate) {
    return true;
  }
  VLOGF(2) << "New bitrate allocation: " << bitrate_allocation.ToString()
           << ", New framerate: " << framerate;

  uint32_t cpb_size_bits = 0;
  base::CheckedNumeric<uint32_t> checked_cpb_size_bits(bitrate);
  checked_cpb_size_bits /= 1000;
  checked_cpb_size_bits *= curr_params_.cpb_window_size_ms;
  if (!checked_cpb_size_bits.AssignIfValid(&cpb_size_bits)) {
    VLOGF(1) << "Too large bitrate: " << bitrate;
    return false;
  }

  const std::optional<uint8_t> level_idc =
      FindValidH265LevelIdc(coded_size_, framerate, bitrate);
  if (!level_idc) {
    VLOGF(1) << "Could not find a valid h265 level for"
             << " bitrate=" << bitrate << " framerate=" << framerate
             << " size=" << coded_size_.ToString();
    return false;
  }

  curr_params_.bitrate_allocation = bitrate_allocation;
  curr_params_.framerate = framerate;
  curr_params_.cpb_size_bits = cpb_size_bits;

  // The bitrate and the framerate are not signaled in the parameter sets, so
  // changing them doesn't require a new IDR picture unless the stream doesn't
  // conform to the current level any longer. The level is never lowered in
  // order not to produce IDR pictures needlessly.
  if (*level_idc > level_idc_) {
    DVLOGF(2) << "Changing general_level_idc from "
              << base::strict_cast<int>(level_idc_) << " to "
              << base::strict_cast<int>(*level_idc);
    level_idc_ = *level_idc;
    UpdateParameterSets();
    // The updated parameter sets have to be sent with an IDR picture.
    encoding_parameters_changed_ = true;
  }

  return true;
}

void H265VaapiVideoEncoderDelegate::UpdateParameterSets() {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);

  vps_.profile_tier_level.general_level_idc = level_idc_;
  sps_.profile_tier_level.general_level_idc = level_idc_;

  packed_parameter_sets_.Reset();
  BuildPackedH265VPS(packed_parameter_sets_, vps_);
  BuildPackedH265SPS(packed_parameter_sets_, sps_);
  BuildPackedH265PPS(packed_parameter_sets_, pps_);
}

void H265VaapiVideoEncoderDelegate::GeneratePackedSliceHeader(
    H26xAnnexBBitstreamBuilder& packed_slice_header,
    const VAEncPictureParameterBufferHEVC& pic_param,
    const VAEncSliceParameterBufferHEVC& slice_param,
    const H265Picture& pic) const {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);

  const bool is_idr = pic_param.pic_fields.bits.idr_pic_flag;
  const bool is_b_slice =
      slice_param.slice_type == H265SliceHeader::kSliceTypeB;
  const bool is_i_slice =
      slice_param.slice_type == H265SliceHeader::kSliceTypeI;

  // The syntax elements whose presence conditions are not fulfilled by the
  // parameter sets generated in Initialize() are not handled here.
  DCHECK(!pps_.dependent_slice_segments_enabled_flag);
  DCHECK(!pps_.output_flag_present_flag);
  DCHECK_EQ(pps_.num_extra_slice_header_bits, 0);
  DCHECK(!pps_.lists_modification_present_flag);
  DCHECK(!pps_.weighted_pred_flag);
  DCHECK(!pps_.weighted_bipred_flag);
  DCHECK(!pps_.tiles_enabled_flag);
  DCHECK(!pps_.entropy_coding_sync_enabled_flag);
  DCHECK(!pps_.slice_segment_header_extension_present_flag);
  DCHECK(!sps_.separate_colour_plane_flag);
  DCHECK(!sps_.long_term_ref_pics_present_flag);
  DCHECK_EQ(sps_.num_short_term_ref_pic_sets, 0);

  // Slice segment header (spec section 7.3.6.1).
  packed_slice_header.BeginNALU(
      static_cast<H265NALU::Type>(pic_param.nal_unit_type));

  packed_slice_header.AppendBits(1, 1);  // first_slice_segment_in_pic_flag
  if (pic_param.nal_unit_type >= H265NALU::BLA_W_LP &&
      pic_param.nal_unit_type <= H265NALU::RSV_IRAP_VCL23) {
    packed_slice_header.AppendBits(
        1, pic_param.pic_fields.bits.no_output_of_prior_pics_flag);
  }
  packed_slice_header.AppendUE(slice_param.slice_pic_parameter_set_id);
  packed_slice_header.AppendUE(slice_param.slice_type);

  if (!is_idr) {
    packed_slice_header.AppendBits(sps_.log2_max_pic_order_cnt_lsb_minus4 + 4,
                                   pic.slice_pic_order_cnt_lsb_);
    // The reference picture set is not one of the SPS.
    packed_slice_header.AppendBits(1, 0);  // short_term_ref_pic_set_sps_flag
    // st_ref_pic_set(num_short_term_ref_pic_sets) (spec section 7.3.7). As
    // its index is zero, inter_ref_pic_set_prediction_flag is not present.
    // The only reference picture is the one in |slice_param.ref_pic_list0|,
    // it precedes |pic| in the output order.
    const int delta_poc =
        pic.pic_order_cnt_val_ - slice_param.ref_pic_list0[0].pic_order_cnt;
    DCHECK_GT(delta_poc, 0);
    packed_slice_header.AppendUE(1);              // num_negative_pics
    packed_slice_header.AppendUE(0);              // num_positive_pics
    packed_slice_header.AppendUE(delta_poc - 1);  // delta_poc_s0_minus1[0]
    packed_slice_header.AppendBits(1, 1);         // used_by_curr_pic_s0_flag[0]
    if (sps_.sps_temporal_mvp_enabled_flag) {
      packed_slice_header.AppendBits(
          1, slice_param.slice_fields.bits.slice_temporal_mvp_enabled_flag);
    }
  }

  if (sps_.sample_adaptive_offset_enabled_flag) {
    packed_slice_header.AppendBits(
        1, slice_param.slice_fields.bits.slice_sao_luma_flag);
    // ChromaArrayType is not zero.
    packed_slice_header.AppendBits(
        1, slice_param.slice_fields.bits.slice_sao_chroma_flag);
  }

  if (!is_i_slice) {
    packed_slice_header.AppendBits(
        1, slice_param.slice_fields.bits.num_ref_idx_active_override_flag);
    if (slice_param.slice_fields.bits.num_ref_idx_active_override_flag) {
      packed_slice_header.AppendUE(slice_param.num_ref_idx_l0_active_minus1);
      if (is_b_slice) {
        packed_slice_header.AppendUE(slice_param.num_ref_idx_l1_active_minus1);
      }
    }
    if (is_b_slice) {
      packed_slice_header.AppendBits(
          1, slice_param.slice_fields.bits.mvd_l1_zero_flag);
    }
    if (pps_.cabac_init_present_flag) {
      packed_slice_header.AppendBits(
          1, slice_param.slice_fields.bits.cabac_init_flag);
    }
    if (slice_param.slice_fields.bits.slice_temporal_mvp_enabled_flag) {
      const bool collocated_from_l0 =
          slice_param.slice_fields.bits.collocated_from_l0_flag;
      if (is_b_slice) {
        packed_slice_header.AppendBits(1, collocated_from_l0);
      }
      if ((collocated_from_l0 &&
           slice_param.num_ref_idx_l0_active_minus1 > 0) ||
          (!collocated_from_l0 &&
           slice_param.num_ref_idx_l1_active_minus1 > 0)) {
        // The collocated picture is the first one of the list.
        packed_slice_header.AppendUE(0);  // collocated_ref_idx
      }
    }
    // five_minus_max_num_merge_cand
    packed_slice_header.AppendUE(kMaxNumMergeCand -
                                 slice_param.max_num_merge_cand);
  }

  packed_slice_header.AppendSE(slice_param.slice_qp_delta);
  if (pps_.pps_slice_chroma_qp_offsets_present_flag) {
    packed_slice_header.AppendSE(slice_param.slice_cb_qp_offset);
    packed_slice_header.AppendSE(slice_param.slice_cr_qp_offset);
  }
  if (pps_.deblocking_filter_override_enabled_flag) {
    packed_slice_header.AppendBits(1, 0);  // deblocking_filter_override_flag
  }
  if (pps_.pps_loop_filter_across_slices_enabled_flag &&
      (slice_param.slice_fields.bits.slice_sao_luma_flag ||
       slice_param.slice_fields.bits.slice_sao_chroma_flag ||
       !slice_param.slice_fields.bits.slice_deblocking_filter_disabled_flag)) {
    packed_slice_header.AppendBits(
        1, slice_param.slice_fields.bits
               .slice_loop_filter_across_slices_enabled_flag);
  }

  // The slice segment header ends with byte_alignment() (spec section 7.3.2.5)
  // whose bit pattern is the same as the one of rbsp_trailing_bits().
  packed_slice_header.FinishNALU();
}

bool H265VaapiVideoEncoderDelegate::SubmitFrameParameters(
    EncodeJob& job,
    H265Picture& pic,
    H265Picture* ref_pic) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);

  const bool is_idr = pic.nal_unit_type_ == H265NALU::IDR_W_RADL;
  DCHECK_EQ(is_idr, !ref_pic);

  const uint32_t bitrate_bps = curr_params_.bitrate_allocation.GetSumBps();

  VAEncSequenceParameterBufferHEVC seq_param = {};
  seq_param.general_profile_idc =
      base::checked_cast<uint8_t>(sps_.profile_tier_level.general_profile_idc);
  seq_param.general_level_idc =
      base::checked_cast<uint8_t>(sps_.profile_tier_level.general_level_idc);
  // Main tier.
  seq_param.general_tier_flag = 0;
  seq_param.intra_period = kIDRPeriod;
  seq_param.intra_idr_period = kIDRPeriod;
  seq_param.ip_period = kIPPeriod;
  seq_param.bits_per_second = bitrate_bps;
  seq_param.pic_width_in_luma_samples =
      base::checked_cast<uint16_t>(sps_.pic_width_in_luma_samples);
  seq_param.pic_height_in_luma_samples =
      base::checked_cast<uint16_t>(sps_.pic_height_in_luma_samples);
#define SPS_TO_SP_FS(a) seq_param.seq_fields.bits.a = sps_.a;
  SPS_TO_SP_FS(chroma_format_idc);
  SPS_TO_SP_FS(separate_colour_plane_flag);
  SPS_TO_SP_FS(bit_depth_luma_minus8);
  SPS_TO_SP_FS(bit_depth_chroma_minus8);
  SPS_TO_SP_FS(scaling_list_enabled_flag);
  SPS_TO_SP_FS(strong_intra_smoothing_enabled_flag);
  SPS_TO_SP_FS(amp_enabled_flag);
  SPS_TO_SP_FS(sample_adaptive_offset_enabled_flag);
  SPS_TO_SP_FS(pcm_enabled_flag);
  SPS_TO_SP_FS(pcm_loop_filter_disabled_flag);
  SPS_TO_SP_FS(sps_temporal_mvp_enabled_flag);
#undef SPS_TO_SP_FS
  seq_param.log2_min_luma_coding_block_size_minus3 =
      base::checked_cast<uint8_t>(sps_.log2_min_luma_coding_block_size_minus3);
  seq_param.log2_diff_max_min_luma_coding_block_size =
      base::checked_cast<uint8_t>(
          sps_.log2_diff_max_min_luma_coding_block_size);
  seq_param.log2_min_transform_block_size_minus2 = base::checked_cast<uint8_t>(
      sps_.log2_min_luma_transform_block_size_minus2);
  seq_param.log2_diff_max_min_transform_block_size =
      base::checked_cast<uint8_t>(
          sps_.log2_diff_max_min_luma_transform_block_size);
  seq_param.max_transform_hierarchy_depth_inter =
      base::checked_cast<uint8_t>(sps_.max_transform_hierarchy_depth_inter);
  seq_param.max_transform_hierarchy_depth_intra =
      base::checked_cast<uint8_t>(sps_.max_transform_hierarchy_depth_intra);
  seq_param.vui_parameters_present_flag = 0;

  VAEncPictureParameterBufferHEVC pic_param = {};
  pic_param.decoded_curr_pic.picture_id =
      pic.AsVaapiH265Picture()->va_surface_id();
  pic_param.decoded_curr_pic.pic_order_cnt = pic.pic_order_cnt_val_;
  pic_param.decoded_curr_pic.flags = 0;
  for (VAPictureHEVC& reference_frame : pic_param.reference_frames) {
    InitVAPictureHEVC(reference_frame);
  }
  if (ref_pic) {
    VAPictureHEVC& reference_frame = pic_param.reference_frames[0];
    reference_frame.picture_id = ref_pic->AsVaapiH265Picture()->va_surface_id();
    reference_frame.pic_order_cnt = ref_pic->pic_order_cnt_val_;
    // The reference picture precedes |pic| in the output order.
    reference_frame.flags = VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE;
  }
  pic_param.coded_buf = job.coded_buffer_id();
  // The collocated picture used for the temporal motion vector prediction is
  // the only reference picture.
  pic_param.collocated_ref_pic_index =
      (ref_pic && sps_.sps_temporal_mvp_enabled_flag) ? 0 : 0xff;
  pic_param.last_picture = 0;
  pic_param.pic_init_qp =
      base::checked_cast<uint8_t>(pps_.init_qp_minus26 + 26);
  pic_param.diff_cu_qp_delta_depth =
      base::checked_cast<uint8_t>(pps_.diff_cu_qp_delta_depth);
  pic_param.pps_cb_qp_offset =
      base::checked_cast<int8_t>(pps_.pps_cb_qp_offset);
  pic_param.pps_cr_qp_offset =
      base::checked_cast<int8_t>(pps_.pps_cr_qp_offset);
  pic_param.log2_parallel_merge_level_minus2 =
      base::checked_cast<uint8_t>(pps_.log2_parallel_merge_level_minus2);
  pic_param.ctu_max_bitsize_allowed = 0;
  pic_param.num_ref_idx_l0_default_active_minus1 =
      base::checked_cast<uint8_t>(pps_.num_ref_idx_l0_default_active_minus1);
  pic_param.num_ref_idx_l1_default_active_minus1 =
      base::checked_cast<uint8_t>(pps_.num_ref_idx_l1_default_active_minus1);
  pic_param.slice_pic_parameter_set_id =
      base::checked_cast<uint8_t>(pps_.pps_pic_parameter_set_id);
  pic_param.nal_unit_type = base::checked_cast<uint8_t>(pic.nal_unit_type_);
  pic_param.pic_fields.bits.idr_pic_flag = is_idr;
  // 1 for an I picture and 2 for a P picture. An inter picture is a P picture
  // even if it is encoded with GPB slices.
  pic_param.pic_fields.bits.coding_type = is_idr ? 1 : 2;
  // Every picture is referenced by the next picture.
  pic_param.pic_fields.bits.reference_pic_flag = 1;
#define PPS_TO_PP_PF(a) pic_param.pic_fields.bits.a = pps_.a;
  PPS_TO_PP_PF(dependent_slice_segments_enabled_flag);
  PPS_TO_PP_PF(sign_data_hiding_enabled_flag);
  PPS_TO_PP_PF(constrained_intra_pred_flag);
  PPS_TO_PP_PF(transform_skip_enabled_flag);
  PPS_TO_PP_PF(cu_qp_delta_enabled_flag);
  PPS_TO_PP_PF(weighted_pred_flag);
  PPS_TO_PP_PF(weighted_bipred_flag);
  PPS_TO_PP_PF(transquant_bypass_enabled_flag);
  PPS_TO_PP_PF(tiles_enabled_flag);
  PPS_TO_PP_PF(entropy_coding_sync_enabled_flag);
  PPS_TO_PP_PF(loop_filter_across_tiles_enabled_flag);
  PPS_TO_PP_PF(pps_loop_filter_across_slices_enabled_flag);
#undef PPS_TO_PP_PF
  pic_param.pic_fields.bits.scaling_list_data_present_flag = 0;
  pic_param.pic_fields.bits.screen_content_flag = 0;
  pic_param.pic_fields.bits.enable_gpu_weighted_prediction = 0;
  pic_param.pic_fields.bits.no_output_of_prior_pics_flag = 0;

  // The whole picture is encoded as a single slice.
  VAEncSliceParameterBufferHEVC slice_param = {};
  slice_param.slice_segment_address = 0;
  slice_param.num_ctu_in_slice = num_ctus_in_pic_;
  if (is_idr) {
    slice_param.slice_type = H265SliceHeader::kSliceTypeI;
  } else {
    slice_param.slice_type =
        use_gpb_ ? H265SliceHeader::kSliceTypeB : H265SliceHeader::kSliceTypeP;
  }
  slice_param.slice_pic_parameter_set_id = pic_param.slice_pic_parameter_set_id;
  slice_param.num_ref_idx_l0_active_minus1 =
      pic_param.num_ref_idx_l0_default_active_minus1;
  slice_param.num_ref_idx_l1_active_minus1 =
      pic_param.num_ref_idx_l1_default_active_minus1;
  for (VAPictureHEVC& va_pic : slice_param.ref_pic_list0) {
    InitVAPictureHEVC(va_pic);
  }
  for (VAPictureHEVC& va_pic : slice_param.ref_pic_list1) {
    InitVAPictureHEVC(va_pic);
  }
  if (ref_pic) {
    slice_param.ref_pic_list0[0] = pic_param.reference_frames[0];
    if (use_gpb_) {
      // Both reference picture lists of a GPB slice are identical.
      slice_param.ref_pic_list1[0] = pic_param.reference_frames[0];
    }
  }
  slice_param.max_num_merge_cand = kMaxNumMergeCand;
  // The QP is determined by the driver rate controller.
  slice_param.slice_qp_delta = 0;
  slice_param.slice_cb_qp_offset = 0;
  slice_param.slice_cr_qp_offset = 0;
  slice_param.slice_beta_offset_div2 = 0;
  slice_param.slice_tc_offset_div2 = 0;
  slice_param.slice_fields.bits.last_slice_of_pic_flag = 1;
  slice_param.slice_fields.bits.dependent_slice_segment_flag = 0;
  slice_param.slice_fields.bits.colour_plane_id = 0;
  slice_param.slice_fields.bits.slice_temporal_mvp_enabled_flag =
      !is_idr && sps_.sps_temporal_mvp_enabled_flag;
  slice_param.slice_fields.bits.slice_sao_luma_flag =
      sps_.sample_adaptive_offset_enabled_flag;
  slice_param.slice_fields.bits.slice_sao_chroma_flag =
      sps_.sample_adaptive_offset_enabled_flag;
  slice_param.slice_fields.bits.num_ref_idx_active_override_flag = 0;
  slice_param.slice_fields.bits.mvd_l1_zero_flag = 0;
  slice_param.slice_fields.bits.cabac_init_flag = 0;
  slice_param.slice_fields.bits.slice_deblocking_filter_disabled_flag = 0;
  slice_param.slice_fields.bits.slice_loop_filter_across_slices_enabled_flag =
      pps_.pps_loop_filter_across_slices_enabled_flag;
  slice_param.slice_fields.bits.collocated_from_l0_flag = !is_idr;

  std::vector<VaapiWrapper::VABufferDescriptor> va_buffers = {
      {VAEncSequenceParameterBufferType, sizeof(seq_param), &seq_param},
      {VAEncPictureParameterBufferType, sizeof(pic_param), &pic_param}};

  std::array<std::vector<uint8_t>, 3> misc_buffers;
  CreateVAEncRateControlParams(
      bitrate_bps, curr_params_.cpb_window_size_ms,
      base::strict_cast<uint32_t>(pic_param.pic_init_qp),
      base::strict_cast<uint32_t>(curr_params_.min_qp),
      base::strict_cast<uint32_t>(curr_params_.max_qp), curr_params_.framerate,
      curr_params_.cpb_size_bits, misc_buffers);
  for (const std::vector<uint8_t>& misc_buffer : misc_buffers) {
    va_buffers.push_back(
        {VAEncMiscParameterBufferType, misc_buffer.size(), misc_buffer.data()});
  }

  // The packed headers are submitted in the order they have in the stream:
  // the parameter sets precede the slice segment.
  VAEncPackedHeaderParameterBuffer packed_parameter_sets_param = {};
  if (is_idr && submit_packed_parameter_sets_) {
    // We always generate VPS, SPS and PPS with an IDR frame. All of them are
    // submitted as a single packed sequence header.
    packed_parameter_sets_param.type = VAEncPackedHeaderSequence;
    packed_parameter_sets_param.bit_length = base::checked_cast<uint32_t>(
        packed_parameter_sets_.BytesInBuffer() * 8);
    packed_parameter_sets_param.has_emulation_bytes = 1;
    va_buffers.push_back({VAEncPackedHeaderParameterBufferType,
                          sizeof(packed_parameter_sets_param),
                          &packed_parameter_sets_param});
    va_buffers.push_back({VAEncPackedHeaderDataBufferType,
                          packed_parameter_sets_.BytesInBuffer(),
                          packed_parameter_sets_.data().data()});
  }

  H26xAnnexBBitstreamBuilder packed_slice_header(
      /*insert_emulation_prevention_bytes=*/true);
  VAEncPackedHeaderParameterBuffer packed_slice_header_param = {};
  if (submit_packed_slice_header_) {
    GeneratePackedSliceHeader(packed_slice_header, pic_param, slice_param, pic);
    packed_slice_header_param.type = VAEncPackedHeaderSlice;
    packed_slice_header_param.bit_length =
        base::checked_cast<uint32_t>(packed_slice_header.BytesInBuffer() * 8);
    packed_slice_header_param.has_emulation_bytes = 1;
    va_buffers.push_back({VAEncPackedHeaderParameterBufferType,
                          sizeof(packed_slice_header_param),
                          &packed_slice_header_param});
    va_buffers.push_back({VAEncPackedHeaderDataBufferType,
                          packed_slice_header.BytesInBuffer(),
                          packed_slice_header.data().data()});
  }

  va_buffers.push_back(
      {VAEncSliceParameterBufferType, sizeof(slice_param), &slice_param});

  return vaapi_wrapper_->SubmitBuffers(va_buffers);
}

}  // namespace media
