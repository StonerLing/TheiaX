// Copyright (C) 2013 The Regents of the University of California (Regents).
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

#include <Eigen/Core>
#include <gflags/gflags.h>

#include <stdio.h>
#include <string>

#include "FreeImagePlus.h"
#include "theia/image/image.h"
#include "theia/util/random.h"
#include "gtest/gtest.h"

DEFINE_string(test_img, "image/test1.jpg", "Name of test image file.");

namespace theia {
namespace {

RandomNumberGenerator rng(51);

std::string img_filename = THEIA_DATA_DIR + std::string("/") + FLAGS_test_img;

#define ASSERT_IMG_EQ(fip_img, theia_img, rows, cols)                      \
  ASSERT_EQ(fip_img.getWidth(), theia_img.Cols());                         \
  ASSERT_EQ(fip_img.getHeight(), theia_img.Rows());                        \
  ASSERT_EQ(fip_img.getBitsPerPixel() / 32, theia_img.Channels());         \
  for (int x = 0; x < cols; ++x) {                                         \
    for (int y = 0; y < rows; ++y) {                                       \
      BYTE* scanline = fip_img.getScanLine(rows - 1 - y);                  \
      float* pixel =                                                       \
          reinterpret_cast<float*>(scanline) + (x * theia_img.Channels()); \
      for (int c = 0; c < theia_img.Channels(); c++) {                     \
        ASSERT_EQ(pixel[c], theia_img.GetXY(x, y, c));                     \
      }                                                                    \
    }                                                                      \
  }
float Interpolate(const FloatImage& image,
                  const double x,
                  const double y,
                  const int c) {
  const float x_fix = x - 0.5;
  const float y_fix = y - 0.5;

  float intpart;
  const float s = std::modf(x_fix, &intpart);
  const int left = static_cast<int>(intpart);
  const float t = std::modf(y_fix, &intpart);
  const int top = static_cast<int>(intpart);

  const float v0 = image.GetXY(left, top, 0);
  const float v1 = image.GetXY(left + 1, top, 0);
  const float v2 = image.GetXY(left, top + 1, 0);
  const float v3 = image.GetXY(left + 1, top + 1, 0);

  return (1.0 - t) * (v0 * (1.0 - s) + v1 * s) + t * (v2 * (1.0 - s) + v3 * s);
}

}  // namespace

// Test that inputting the old fashioned way is the same as through our class.
TEST(Image, RGBInput) {
  fipImage fip_img;
  fip_img.load(img_filename.c_str());
  fip_img.convertToRGBF();

  FloatImage theia_img(img_filename);

  int rows = fip_img.getHeight();
  int cols = fip_img.getWidth();

  // Assert each pixel value is exactly the same!
  ASSERT_IMG_EQ(fip_img, theia_img, rows, cols);
}

// Test that width and height methods work.
TEST(Image, RGBColsRows) {
  fipImage fip_img;
  fip_img.load(img_filename.c_str());
  FloatImage theia_img(img_filename);

  int true_height = fip_img.getHeight();
  int true_width = fip_img.getWidth();

  ASSERT_EQ(theia_img.Cols(), true_width);
  ASSERT_EQ(theia_img.Rows(), true_height);
}

// Test that inputting the old fashioned way is the same as through our class.
TEST(Image, ConvertToGrayscaleImage) {}

TEST(Image, ConvertToRGBImage) {}

TEST(Image, BillinearInterpolate) {}

TEST(Image, ScalePixels) {}

TEST(Image, Resize) {}

TEST(Image, ResizeUninitialized) {}

}  // namespace theia
