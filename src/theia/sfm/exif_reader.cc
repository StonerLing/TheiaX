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

#include "theia/sfm/exif_reader.h"

#include <FreeImagePlus.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <fstream>   // NOLINT
#include <iostream>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// To avoid the same name defination in FreeImagePlus.h/Windows.h
#undef max
#undef min

#include "theia/image/image.h"
#include "theia/sfm/camera_intrinsics_prior.h"
#include "theia/util/map_util.h"

// Generated file
#include "camera_sensor_database.h"

namespace theia {
namespace {

void RemoveLeadingTrailingSpaces(std::string* str) {
  size_t p = str->find_first_not_of(" \t");
  str->erase(0, p);

  p = str->find_last_not_of(" \t");
  if (std::string::npos != p) str->erase(p + 1);
}

std::string ToLowercase(const std::string& str) {
  std::string str2 = str;
  RemoveLeadingTrailingSpaces(&str2);
  std::transform(str2.begin(), str2.end(), str2.begin(), ::tolower);
  if (!str2.empty() &&
      (str2[str2.size() - 1] == '\r' || str2[str2.size() - 1] == '\0')) {
    str2.erase(str2.size() - 1);
  }
  return str2;
}

std::vector<std::string> SplitString(const std::string& s, const char delim) {
  std::vector<std::string> tokens;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, delim)) {
    tokens.push_back(token);
  }
  return tokens;
}

bool IsValidFocalLength(const double focal_length) {
  return std::isfinite(focal_length) && focal_length > 0;
}

// Find the specific key of EXIF metadata, if not found than return nullptr.
auto ExifFindOrNull(const Exiv2::ExifData& exif_data, std::string_view key) {
  Exiv2::ExifKey exif_key(key.data());
  auto iter = exif_data.findKey(exif_key);
  if (iter == exif_data.end()) {
    LOG(WARNING) << "Could not find key: " << key;
    return Exiv2::Value::UniquePtr(nullptr);
  }
  return iter->getValue();
}

// Find the specific key of EXIF metadata, if not found than export an error.
auto ExifFindOrDie(const Exiv2::ExifData& exif_data, std::string_view key)
    -> Exiv2::Value::UniquePtr {
  Exiv2::ExifKey exif_key(key.data());
  auto iter = exif_data.findKey(exif_key);
  CHECK(iter != exif_data.end()) << "Could not find key: " << key;
  return iter->getValue();
}

// Find the specific key of EXIF metadata, if not found than return a default
// value.
auto ExifFindOrDefault(const Exiv2::ExifData& exif_data,
                       std::string_view key,
                       float default_value) -> Exiv2::Value::UniquePtr {
  Exiv2::ExifKey exif_key(key.data());
  auto iter = exif_data.findKey(exif_key);
  if (iter == exif_data.end()) {
    LOG(WARNING) << "Could not find key: " << key
                 << ", default value will be set: " << default_value;
    return Exiv2::Value::UniquePtr(new Exiv2::ValueType<float>(default_value));
  }
  return iter->getValue();
}

// Since image might be cropped, we need to get the image size from raw data
// rather than EXIF. Reaturs image size as [imageWidth, imageHeight].
std::pair<int, int> GetSizeFromImageData(std::string_view image_path) {
  fipImage image;
  image.load(image_path.data());
  return std::make_pair(image.getWidth(), image.getHeight());
}

enum ExifFocalPlaneResolutionUnit {
  NONE = 1,
  INCHES = 2,
  CM = 3,
  MM = 4,
  UM = 5
};

constexpr double kMillimetersPerInch = 25.4;
constexpr double kMillimetersPerCentimeter = 10.0;
constexpr double kMillimetersPerMicron = 1.0 / 1000.0;

}  // namespace

ExifReader::ExifReader() { LoadSensorWidthDatabase(); }

void ExifReader::LoadSensorWidthDatabase() {
  std::stringstream ifs(camera_sensor_database_txt, std::ios::in);

  while (!ifs.eof()) {
    // Read in the filename.
    std::string line;
    std::getline(ifs, line);
    if (line.size() == 0) {
      break;
    }

    const auto& tokens = SplitString(line, ';');
    CHECK_EQ(tokens.size(), 3);

    const std::string make = ToLowercase(tokens[0]);
    const std::string model = ToLowercase(tokens[1]);
    const double camera_sensor_width = stod(tokens[2]);

    // In the database, the model includes the make.
    InsertOrDie(&sensor_width_database_, model, camera_sensor_width);
  }
}

bool ExifReader::ExtractEXIFMetadata(
    const std::string& image_file,
    CameraIntrinsicsPrior* camera_intrinsics_prior) const {
  CHECK_NOTNULL(camera_intrinsics_prior);

  auto image = Exiv2::ImageFactory::open(image_file.c_str());
  CHECK_NOTNULL(image.get())->readMetadata();
  Exiv2::ExifData& exif_data = image->exifData();

  // Set the image dimensions.
  auto [image_width, image_height] = GetSizeFromImageData(image_file);
  camera_intrinsics_prior->image_width = image_width;
  camera_intrinsics_prior->image_height = image_height;

  // Set principal point.
  camera_intrinsics_prior->principal_point.is_set = true;
  camera_intrinsics_prior->principal_point.value[0] =
      camera_intrinsics_prior->image_width / 2.0;
  camera_intrinsics_prior->principal_point.value[1] =
      camera_intrinsics_prior->image_height / 2.0;

  // Attempt to set the focal length from the plane resolution, then try the
  // sensor width database if that fails.
  if (!setFocalLengthPriorFromExifResolution(exif_data,
                                             camera_intrinsics_prior) &&
      !SetFocalLengthFromSensorDatabase(exif_data, camera_intrinsics_prior) &&
      !setFocalLengthPriorFromExif35Film(exif_data, camera_intrinsics_prior)) {
    return true;
  }

  // If we passed the if statement above, then we know that the focal length
  // gathered from EXIF is valid and so we set the camera intrinsics prior for
  // focal lengths to true.
  camera_intrinsics_prior->focal_length.is_set = true;

  // Set GPS
  if (!setGpsPriorFromExif(exif_data, camera_intrinsics_prior)) {
    LOG(WARNING) << "Could not set GPS prior: " << image_file;
  }

  return true;
}

bool ExifReader::setFocalLengthPriorFromExifResolution(
    const Exiv2::ExifData& exif_data,
    CameraIntrinsicsPrior* camera_intrinsics_prior) const {
  float kMinFocalLength = 1.e-2f;
  float focal_length_mm =
      ExifFindOrDefault(exif_data, "Exif.Photo.FocalLength", kMinFocalLength)
          ->toFloat();
  Exiv2::Value::UniquePtr value_ptr =
      ExifFindOrNull(exif_data, "Exif.Image.FocalPlaneXResolution");
  if (value_ptr == nullptr) return false;
  float focal_plane_x_pixels_per_unit = value_ptr->toFloat();

  value_ptr = ExifFindOrNull(exif_data, "Exif.Image.FocalPlaneYResolution");
  if (value_ptr == nullptr) return false;
  float focal_plane_y_pixels_per_unit = value_ptr->toFloat();

  value_ptr = ExifFindOrNull(exif_data, "Exif.Image.FocalPlaneResolutionUnit");
  if (value_ptr == nullptr) return false;
  int focal_plane_resolution_unit = static_cast<int>(value_ptr->toInt64());

  if (focal_length_mm < kMinFocalLength || focal_plane_x_pixels_per_unit <= 0 ||
      focal_plane_y_pixels_per_unit <= 0) {
    return false;
  }

  // CCD resolution is the pixels per unit resolution(mm) of CCD
  double ccd_mm_per_unit = 1.0;
  switch (focal_plane_resolution_unit) {
    case ExifFocalPlaneResolutionUnit::INCHES:
      ccd_mm_per_unit = kMillimetersPerInch;
      break;
    case ExifFocalPlaneResolutionUnit::CM:
      ccd_mm_per_unit = kMillimetersPerCentimeter;
      break;
    case ExifFocalPlaneResolutionUnit::MM:
      break;
    case ExifFocalPlaneResolutionUnit::UM:
      ccd_mm_per_unit = kMillimetersPerMicron;
      break;
    default:
      LOG(WARNING) << "Undefined resolution unit from EXIF meta data.";
      return false;
      break;
  }

  // Gets ccd size in mm
  int captured_image_width_pixels = static_cast<int>(
      ExifFindOrDie(exif_data, "Exif.Photo.PixelXDimension")->toInt64());
  int captured_image_height_pixels = static_cast<int>(
      ExifFindOrDie(exif_data, "Exif.Photo.PixelYDimension")->toInt64());

  double ccd_width_mm =
      captured_image_width_pixels /
      (static_cast<double>(focal_plane_x_pixels_per_unit) / ccd_mm_per_unit);
  double ccd_height_mm =
      captured_image_height_pixels /
      (static_cast<double>(focal_plane_y_pixels_per_unit) / ccd_mm_per_unit);

  int stored_image_width_pixels = camera_intrinsics_prior->image_width;
  int stored_image_height_pixels = camera_intrinsics_prior->image_height;

  double focal_length_x_pixels = static_cast<double>(focal_length_mm) *
                                 stored_image_width_pixels / ccd_width_mm;
  double focal_length_y_pixels = static_cast<double>(focal_length_mm) *
                                 stored_image_height_pixels / ccd_height_mm;

  // Final focal length in pixels should belong to (0, +\infinite)
  double focal_length_pixels =
      (focal_length_x_pixels + focal_length_y_pixels) / 2;
  camera_intrinsics_prior->focal_length.value[0] = focal_length_pixels;
  return IsValidFocalLength(focal_length_pixels);
}

bool ExifReader::SetFocalLengthFromSensorDatabase(
    const Exiv2::ExifData& exif_data,
    CameraIntrinsicsPrior* camera_intrinsics_prior) const {
  if (sensor_width_database_.empty()) {
    LOG(WARNING) << "The camera sensor width database is empty";
    return false;
  }

  int image_width_pixels = camera_intrinsics_prior->image_width;
  int image_height_pixels = camera_intrinsics_prior->image_height;

  float kMinFocalLength = 1.e-2f;
  double focal_length_mm = static_cast<double>(
      ExifFindOrDefault(exif_data, "Exif.Photo.FocalLength", kMinFocalLength)
          ->toFloat());

  // Get sensor name from EXIF metadata
  Exiv2::Value::UniquePtr value_ptr =
      ExifFindOrNull(exif_data, "Exif.Image.Make");
  if (value_ptr == nullptr) return false;
  std::string make = value_ptr->toString();

  value_ptr = ExifFindOrNull(exif_data, "Exif.Image.Model");
  if (value_ptr == nullptr) return false;
  std::string model = value_ptr->toString();
  std::string sensor_name = ToLowercase(make) + " " + ToLowercase(model);

  // Try to find the sensor infomation in database
  auto sensor_width_ptr = FindOrNull(sensor_width_database_, sensor_name);
  if (sensor_width_ptr == nullptr) {
    LOG(WARNING) << "Could not find the sensor infomation in database: "
                 << sensor_name;
    return false;
  }
  double sensor_width_mm = *sensor_width_ptr;
  LOG(INFO) << "Sensor width = " << sensor_width_mm;
  if (sensor_width_mm <= 0) return false;

  double max_image_dimension_pixels =
      static_cast<double>(std::max(image_width_pixels, image_height_pixels));
  double focal_length_pixels =
      max_image_dimension_pixels * focal_length_mm / sensor_width_mm;

  camera_intrinsics_prior->focal_length.value[0] = focal_length_pixels;
  return IsValidFocalLength(focal_length_pixels);
}

bool ExifReader::setFocalLengthPriorFromExif35Film(
    const Exiv2::ExifData& exif_data,
    CameraIntrinsicsPrior* camera_intrinsics_prior) const {
  auto value_ptr =
      ExifFindOrNull(exif_data, "Exif.Photo.FocalLengthIn35mmFilm");
  if (value_ptr == nullptr) return false;
  double focal_length_in_35mm_file = static_cast<double>(value_ptr->toFloat());

  int image_width = camera_intrinsics_prior->image_width;
  int image_height = camera_intrinsics_prior->image_height;

  double focal_length =
      std::max(image_width, image_height) * focal_length_in_35mm_file / 36.0;

  camera_intrinsics_prior->focal_length.value[0] = focal_length;
  return IsValidFocalLength(focal_length);
}

bool ExifReader::setGpsPriorFromExif(
    const Exiv2::ExifData& exif_data,
    CameraIntrinsicsPrior* camera_intrinsics_prior) const {
  // GPS altitude
  auto altitude_ptr = ExifFindOrNull(exif_data, "Exif.GPSInfo.GPSAltitude");
  if (altitude_ptr != nullptr) {
    camera_intrinsics_prior->altitude.value[0] =
        static_cast<double>(altitude_ptr->toFloat());
    if (ExifFindOrDie(exif_data, "Exif.GPSInfo.GPSAltitudeRef")->toString() ==
        "1") {
      camera_intrinsics_prior->altitude.value[0] *= -1;
    }
  }

  // GPS latitude
  auto latitude_ptr = ExifFindOrNull(exif_data, "Exif.GPSInfo.GPSLatitude");
  if (latitude_ptr == nullptr) return false;
  camera_intrinsics_prior->latitude.value[0] =
      static_cast<double>(latitude_ptr->toFloat(0)) +
      static_cast<double>(latitude_ptr->toFloat(1)) / 60.0 +
      static_cast<double>(latitude_ptr->toFloat(2)) / 3600.0;
  if (ExifFindOrDie(exif_data, "Exif.GPSInfo.GPSLatitudeRef")->toString() ==
      "S") {
    camera_intrinsics_prior->latitude.value[0] *= -1;
  }

  // GPS longitude
  auto longitude_ptr = ExifFindOrNull(exif_data, "Exif.GPSInfo.GPSLongitude");
  if (longitude_ptr == nullptr) return false;
  camera_intrinsics_prior->longitude.value[0] =
      static_cast<double>(longitude_ptr->toFloat(0)) +
      static_cast<double>(longitude_ptr->toFloat(1)) / 60.0 +
      static_cast<double>(longitude_ptr->toFloat(2)) / 3600.0;
  if (ExifFindOrDie(exif_data, "Exif.GPSInfo.GPSLongitudeRef")->toString() ==
      "W") {
    camera_intrinsics_prior->longitude.value[0] *= -1;
  }

  return true;
}
}  // namespace theia
