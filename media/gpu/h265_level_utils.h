// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MEDIA_GPU_H265_LEVEL_UTILS_H_
#define MEDIA_GPU_H265_LEVEL_UTILS_H_

#include <stdint.h>

#include <optional>

#include "media/gpu/media_gpu_export.h"
#include "ui/gfx/geometry/size.h"

namespace media {

// Returns the lowest H.265 general_level_idc (30 * level number, e.g. 120 for
// level 4.0) whose Main tier limits from Rec. ITU-T H.265 Tables A.8 and A.9
// are satisfied by a stream of |coded_size| luma samples per picture at
// |framerate| pictures per second and |bitrate_bps| bits per second. Returns
// std::nullopt if |coded_size| is empty or if even level 6.2 is not enough.
MEDIA_GPU_EXPORT std::optional<uint8_t> FindValidH265LevelIdc(
    const gfx::Size& coded_size,
    uint32_t framerate,
    uint32_t bitrate_bps);

}  // namespace media

#endif  // MEDIA_GPU_H265_LEVEL_UTILS_H_
