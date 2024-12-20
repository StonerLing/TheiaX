// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef THEIA_SFM_EXIF_READER_H_
#define THEIA_SFM_EXIF_READER_H_

#include <string>
#include <unordered_map>

#include <exiv2/exiv2.hpp>

#include "theia/util/hash.h"
#include "theia/util/util.h"

namespace theia {

struct CameraIntrinsicsPrior;

// The focal length is set from the EXIF data. We attempt to set the focal
// length from the EXIF focal length plane resolutions. If the plane resolutions
// are not available then we use a sensor database to determine the sensor
// width.
//
// NOTE: Reading in the sensor database is an expensive operation. When
// calibrating many cameras it is best to create a single ExifReader object to
// read the EXIF information of all cameras. This way, the database is only
// loaded once.
class ExifReader {
 public:
  ExifReader();

  // Extracts EXIF metadata from the image file and populates the intrinsics
  // prior object. If the file could not be opened then the function returns
  // false. If no EXIF data is found in the image, then it will be a valid
  // CameraIntrinsicsPrior object with the is_set field set to false for all
  // metadata field. The function will return true in this case.
  bool ExtractEXIFMetadata(
      const std::string& image_file,
      CameraIntrinsicsPrior* camera_intrinsics_prior) const;

 private:
  void LoadSensorWidthDatabase();

  // Set the focal length prior from EXIF FocalPlaneResolution. We esitimate the
  // focal length in pixels via:
  // f_x=f_y=f \times 0.5(width + height) / ccdWidth
  // ccdWidth = width / (xResolution / unit)
  bool setFocalLengthPriorFromExifResolution(
      const Exiv2::ExifData& exif_data,
      CameraIntrinsicsPrior* camera_intrinsics_prior) const;

  // Sets the focal length from a look up in the sensor width database. Returns
  // true if a valid focal length is found and false otherwise.
  bool SetFocalLengthFromSensorDatabase(
      const Exiv2::ExifData& exif_data,
      CameraIntrinsicsPrior* camera_intrinsics_prior) const;

  // Set the focal length prior from camera EXIF FocalLengthIn35mmFilm. We
  // estimate the focal length in pixels via(since the film size is 36\times
  // 24mm):
  // f_x=f_y=f \times 36 / focalLengthIn35mmFilm
  bool setFocalLengthPriorFromExif35Film(
      const Exiv2::ExifData& exif_data,
      CameraIntrinsicsPrior* camera_intrinsics_prior) const;

  // Set the GPS prior from EXIF metadata.
  bool setGpsPriorFromExif(
      const Exiv2::ExifData& exif_data,
      CameraIntrinsicsPrior* camera_intrinsics_prior) const;

  std::unordered_map<std::string, double> sensor_width_database_;

  DISALLOW_COPY_AND_ASSIGN(ExifReader);
};

}  // namespace theia

#endif  // THEIA_SFM_EXIF_READER_H_
