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

#include "theia/image/image.h"

#include <Eigen/Core>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "theia/math/util.h"

namespace {
constexpr static int kFloatBitDepth = 32;  // sizeof(float) * 8

FREE_IMAGE_TYPE ChannelsToImageType(int channels) {
  FREE_IMAGE_TYPE image_type;
  switch (channels) {
    case 1:
      image_type = FIT_FLOAT;
      break;
    case 3:
      image_type = FIT_RGBF;
      break;
    default:
      LOG(ERROR) << "Only support channels equal to 1 or 3";
  }
  return image_type;
}

float* GetXYData(fipImage& image, int x, int y) {
  const int channels = image.getBitsPerPixel() / kFloatBitDepth;
  BYTE* scanline = image.getScanLine(image.getHeight() - 1 - y);
  float* pixel = reinterpret_cast<float*>(scanline) + (x * channels);
  return pixel;
}

const float* GetXYData(const fipImage& image, int x, int y) {
  const int channels = image.getBitsPerPixel() / kFloatBitDepth;
  BYTE* scanline = image.getScanLine(image.getHeight() - 1 - y);
  const float* pixel = reinterpret_cast<float*>(scanline) + (x * channels);
  return pixel;
}

std::vector<float> GenerateGaussianKernel(int kernel_size, float sigma) {
  int half_size = kernel_size / 2;
  std::vector<float> kernel(kernel_size);
  float sum = 0.0f;
  float coefficient = 1.0f / (std::sqrt(2.0f * M_PI) * sigma);

  for (int i = -half_size; i <= half_size; ++i) {
    kernel[i + half_size] =
        coefficient * std::exp(-0.5f * (i * i) / (sigma * sigma));
    sum += kernel[i + half_size];
  }

  for (int i = 0; i < kernel_size; ++i) {
    kernel[i] /= sum;
  }

  return kernel;
}
}  // namespace

namespace theia {

FloatImage::FloatImage() : FloatImage(0, 0, 1) {}

FloatImage ::~FloatImage() {}

// Read from file.
FloatImage::FloatImage(const std::string& filename) { Read(filename); }

FloatImage::FloatImage(const FloatImage& other) { image_ = other.image_; }

FloatImage::FloatImage(const int width, const int height, const int channels)
    : image_(fipImage(ChannelsToImageType(channels),
                      width,
                      height,
                      kFloatBitDepth * channels)) {}

FloatImage::FloatImage(const int width,
                       const int height,
                       const int channels,
                       float* buffer)
    : image_(fipImage(ChannelsToImageType(channels),
                      width,
                      height,
                      kFloatBitDepth * channels)) {
  fipMemoryIO memory_buffer(reinterpret_cast<BYTE*>(buffer));
  CHECK(image_.loadFromMemory(memory_buffer));
}

FloatImage::FloatImage(const fipImage& other) { image_ = fipImage(other); }

FloatImage& FloatImage::operator=(const FloatImage& other) {
  image_ = other.image_;
  return *this;
}

FloatImage::FloatImage(FloatImage&& other) : image_(std::move(other.image_)) {}

FloatImage& FloatImage::operator=(FloatImage&& other) {
  image_ = std::move(other.image_);
  return *this;
}

fipImage& FloatImage::GetFipImage() { return image_; }

const fipImage& FloatImage::GetFipImage() const { return image_; }

int FloatImage::Rows() const { return Height(); }

int FloatImage::Cols() const { return Width(); }

int FloatImage::Width() const { return image_.getWidth(); }

int FloatImage::Height() const { return image_.getHeight(); }

int FloatImage::Channels() const {
  return image_.getBitsPerPixel() / kFloatBitDepth;
}

void FloatImage::SetXY(const int x,
                       const int y,
                       const int c,
                       const float value) {
  DCHECK_GE(x, 0);
  DCHECK_LT(x, Width());
  DCHECK_GE(y, 0);
  DCHECK_LT(y, Height());
  DCHECK_GE(c, 0);
  DCHECK_LT(c, Channels());

  float* pixel = GetXYData(image_, x, y);
  pixel[c] = value;
}

void FloatImage::SetXY(const int x, const int y, const Eigen::Vector3f& rgb) {
  DCHECK_GE(x, 0);
  DCHECK_LT(x, Width());
  DCHECK_GE(y, 0);
  DCHECK_LT(y, Height());
  DCHECK(Channels() == 3 || Channels() == 4);

  float* pixel = GetXYData(image_, x, y);

  pixel[0] = rgb[0];
  pixel[1] = rgb[1];
  pixel[2] = rgb[2];
}

float FloatImage::GetXY(const int x, const int y, const int c) const {
  DCHECK_GE(x, 0);
  DCHECK_LT(x, Width());
  DCHECK_GE(y, 0);
  DCHECK_LT(y, Height());
  DCHECK_GE(c, 0);
  DCHECK_LT(c, Channels());
  return GetXYData(image_, x, y)[c];
}

Eigen::Vector3f FloatImage::GetXY(const int x, const int y) const {
  DCHECK_GE(x, 0);
  DCHECK_LT(x, Width());
  DCHECK_GE(y, 0);
  DCHECK_LT(y, Height());
  const float* pixel = GetXYData(image_, x, y);
  Eigen::Vector3f rgb{pixel[0], pixel[1], pixel[2]};
  return rgb;
}

void FloatImage::SetRowCol(const int row,
                           const int col,
                           const int channel,
                           const float value) {
  SetXY(col, row, channel, value);
}

void FloatImage::SetRowCol(const int row,
                           const int col,
                           const Eigen::Vector3f& rgb) {
  SetXY(col, row, rgb);
}

float FloatImage::GetRowCol(const int row,
                            const int col,
                            const int channel) const {
  return GetXY(col, row, channel);
}

Eigen::Vector3f FloatImage::GetRowCol(const int row, const int col) const {
  return GetXY(col, row);
}

float FloatImage::BilinearInterpolate(const double x,
                                      const double y,
                                      const int c) const {
  DCHECK_GE(c, 0);
  DCHECK_LT(c, Channels());

  int x0 = static_cast<int>(std::floor(x));
  int y0 = static_cast<int>(std::floor(y));

  double dx = x - x0;
  double dy = y - y0;

  const float* pixel00 = GetXYData(image_, x0, y0);          // (x0, y0)
  const float* pixel10 = GetXYData(image_, x0 + 1, y0);      // (x1, y0)
  const float* pixel01 = GetXYData(image_, x0, y0 + 1);      // (x0, y1)
  const float* pixel11 = GetXYData(image_, x0 + 1, y0 + 1);  // (x1, y1)

  float interpolated_value =
      (1 - dy) * ((1 - dx) * pixel00[c] + dx * pixel10[c]) +
      dy * ((1 - dx) * pixel01[c] + dx * pixel11[c]);

  return interpolated_value;
}

Eigen::Vector3f FloatImage::BilinearInterpolate(const double x,
                                                const double y) const {
  Eigen::Vector3f interpolated_pixel{BilinearInterpolate(x, y, 0),
                                     BilinearInterpolate(x, y, 1),
                                     BilinearInterpolate(x, y, 2)};
  return interpolated_pixel;
}

void FloatImage::ConvertToGrayscaleImage() {
  if (Channels() == 1) {
    VLOG(2) << "Image is already a grayscale image. No conversion necessary.";
    return;
  }
  fipImage gray_image(FIT_FLOAT, Width(), Height(), kFloatBitDepth);

  for (int y = 0; y < Height(); ++y) {
    for (int x = 0; x < Width(); ++x) {
      const float* pixel = GetXYData(image_, x, y);

      float gray = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];

      GetXYData(gray_image, x, y)[0] = gray;
    }
  }

  image_ = gray_image;
}

void FloatImage::ConvertToRGBImage() {
  if (Channels() == 3) {
    VLOG(2) << "Image is already an RGB image. No conversion necessary.";
    return;
  }

  if (Channels() == 1) {
    fipImage rgb_image(FIT_FLOAT, Width(), Height(), kFloatBitDepth * 3);
    for (int y = 0; y < Height(); ++y) {
      for (int x = 0; x < Width(); ++x) {
        const float* gray_pixel = GetXYData(image_, x, y);
        float gray = gray_pixel[0];

        float* rgb_pixel = GetXYData(rgb_image, x, y);
        rgb_pixel[0] = gray;  // R
        rgb_pixel[1] = gray;  // G
        rgb_pixel[2] = gray;  // B
      }
    }

    image_ = rgb_image;
  } else {
    LOG(FATAL) << "Converting from " << Channels()
               << " channels to RGB is unsupported.";
  }
}

FloatImage FloatImage::AsGrayscaleImage() const {
  if (Channels() == 1) {
    VLOG(2) << "Image is already a grayscale image. No conversion necessary.";
    return *this;
  }
  FloatImage gray_image(*this);
  gray_image.ConvertToGrayscaleImage();
  return gray_image;
}

FloatImage FloatImage::AsRGBImage() const {
  if (Channels() == 3) {
    VLOG(2) << "Image is already an RGB image. No conversion necessary.";
    return *this;
  }

  FloatImage rgb_image(*this);
  rgb_image.ConvertToRGBImage();
  return rgb_image;
}

void FloatImage::ScalePixels(float scale) {
  for (int y = 0; y < Height(); ++y) {
    for (int x = 0; x < Width(); ++x) {
      float* pixel = GetXYData(image_, x, y);
      for (int c = 0; c < Channels(); ++c) {
        pixel[c] *= scale;
      }
    }
  }
}

void FloatImage::Read(const std::string& filename) {
  CHECK(image_.load(filename.c_str()));
  if (image_.getBitsPerPixel() == 8) {
    image_.convertToFloat();
  } else {
    image_.convertToRGBF();
  }
}

void FloatImage::Write(const std::string& filename) const {
  CHECK(const_cast<fipImage&>(image_).save(filename.c_str()));
}

float* FloatImage::LineData(size_t line) {
  return reinterpret_cast<float*>(image_.getScanLine(line));
}
const float* FloatImage::LineData(size_t line) const {
  return reinterpret_cast<float*>(image_.getScanLine(line));
}

std::vector<float> FloatImage::AsArray(bool is_row_major) const {
  std::vector<float> array(Width() * Height() * Channels());
  size_t i = 0;
  if (is_row_major) {
    for (int y = 0; y < Height(); ++y) {
      BYTE* line = image_.getScanLine(Height() - 1 - y);
      for (int x = 0; x < Width(); ++x) {
        for (int d = 0; d < Channels(); ++d) {
          array[i] = reinterpret_cast<float*>(line)[x * Channels() + d];
          i += 1;
        }
      }
    }
  } else {
    for (int d = 0; d < Channels(); ++d) {
      for (int x = 0; x < Width(); ++x) {
        for (int y = 0; y < Height(); ++y) {
          BYTE* line = image_.getScanLine(Height() - 1 - y);
          array[i] = reinterpret_cast<float*>(line)[x * Channels() + d];
          i += 1;
        }
      }
    }
  }

  return array;
}

FloatImage FloatImage::ComputeGradientX() const {
  float sobel_filter_x[9] = {-.125, 0, .125, -.25, 0, .25, -.125, 0, .125};

  fipImage gradient_x(image_);

  for (int y = 1; y < Height() - 1; ++y) {
    for (int x = 1; x < Width() - 1; ++x) {
      Eigen::Vector3f gradient = Eigen::Vector3f::Zero();

      for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
          const float* pixel = GetXYData(image_, x + i, y + j);
          for (int c = 0; c < Channels(); ++c) {
            gradient[c] += sobel_filter_x[(j + 1) * 3 + (i + 1)] * pixel[c];
          }
        }
      }
      for (int c = 0; c < Channels(); ++c) {
        GetXYData(gradient_x, x, y)[c] = gradient[c];
      }
    }
  }

  return FloatImage(gradient_x);
}

FloatImage FloatImage::ComputeGradientY() const {
  float sobel_filter_y[9] = {-.125, -.25, -.125, 0, 0, 0, .125, .25, .125};
  fipImage gradient_y(image_);

  for (int y = 1; y < Height() - 1; ++y) {
    for (int x = 1; x < Width() - 1; ++x) {
      Eigen::Vector3f gradient = Eigen::Vector3f::Zero();

      for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
          const float* pixel = GetXYData(image_, x + i, y + j);
          for (int c = 0; c < Channels(); ++c) {
            gradient[c] += sobel_filter_y[(j + 1) * 3 + (i + 1)] * pixel[c];
          }
        }
      }

      for (int c = 0; c < Channels(); ++c) {
        GetXYData(gradient_y, x, y)[c] = gradient[c];
      }
    }
  }

  return FloatImage(gradient_y);
}

FloatImage FloatImage::ComputeGradient() const {
  float sobel_filter_x[9] = {-.125, 0, .125, -.25, 0, .25, -.125, 0, .125};
  float sobel_filter_y[9] = {-.125, -.25, -.125, 0, 0, 0, .125, .25, .125};

  fipImage gradient_magnitude(image_);

  for (int y = 0; y < Height(); ++y) {
    for (int x = 0; x < Width(); ++x) {
      Eigen::Vector3f local_gradient_x = Eigen::Vector3f::Zero();
      Eigen::Vector3f local_gradient_y = Eigen::Vector3f::Zero();
      for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
          const float* pixel = GetXYData(image_, x + i, y + j);
          for (int c = 0; c < Channels(); ++c) {
            local_gradient_x[c] +=
                sobel_filter_x[(j + 1) * 3 + (i + 1)] * pixel[c];
            local_gradient_y[c] +=
                sobel_filter_y[(j + 1) * 3 + (i + 1)] * pixel[c];
          }
        }
      }

      Eigen::Vector3f magnitude = Eigen::Vector3f::Zero();
      for (int c = 0; c < Channels(); ++c) {
        magnitude[c] += local_gradient_x[c] * local_gradient_x[c] +
                        local_gradient_y[c] * local_gradient_y[c];
      }
      magnitude.cwiseSqrt();

      for (int c = 0; c < Channels(); ++c) {
        GetXYData(gradient_magnitude, x, y)[c] = magnitude[c];
      }
    }
  }

  return FloatImage(gradient_magnitude);
}

void FloatImage::ApproximateGaussianBlur(const int kernel_size) {
  const int width = Width();
  const int height = Height();
  const int channels = Channels();

  float sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8;
  std::vector<float> gaussian_kernel =
      GenerateGaussianKernel(kernel_size, sigma);

  fipImage blurred_image(image_);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      Eigen::Vector3f blurred_pixel = Eigen::Vector3f::Zero();

      for (int k = -kernel_size / 2; k <= kernel_size / 2; ++k) {
        int x_offset = x + k;
        if (x_offset < 0) x_offset = 0;
        if (x_offset >= width) x_offset = width - 1;

        const float* pixel = GetXYData(image_, x_offset, y);
        float weight = gaussian_kernel[k + kernel_size / 2];

        for (int c = 0; c < channels; ++c) {
          blurred_pixel[c] += weight * pixel[c];
        }
      }

      for (int c = 0; c < channels; ++c) {
        GetXYData(blurred_image, x, y)[c] = blurred_pixel[c];
      }
    }
  }

  fipImage final_blurred_image(blurred_image);
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      Eigen::Vector3f blurred_pixel = Eigen::Vector3f::Zero();

      for (int k = -kernel_size / 2; k <= kernel_size / 2; ++k) {
        int y_offset = y + k;
        if (y_offset < 0) y_offset = 0;
        if (y_offset >= height) y_offset = height - 1;

        const float* pixel = GetXYData(blurred_image, x, y_offset);
        float weight = gaussian_kernel[k + kernel_size / 2];

        for (int c = 0; c < channels; ++c) {
          blurred_pixel[c] += weight * pixel[c];
        }
      }

      for (int c = 0; c < channels; ++c) {
        GetXYData(final_blurred_image, x, y)[c] = blurred_pixel[c];
      }
    }
  }

  image_ = final_blurred_image;
}

void FloatImage::MedianFilter(const int patch_width) {
  const int width = Width();
  const int height = Height();
  const int channels = Channels();

  fipImage filtered_image(image_);

  int patch_radius = patch_width / 2;

  for (int y = patch_radius; y < height - patch_radius; ++y) {
    for (int x = patch_radius; x < width - patch_radius; ++x) {
      std::vector<float> neighborhood;

      for (int j = -patch_radius; j <= patch_radius; ++j) {
        for (int i = -patch_radius; i <= patch_radius; ++i) {
          const float* pixel = GetXYData(image_, x + i, y + j);
          for (int c = 0; c < channels; ++c) {
            neighborhood.push_back(pixel[c]);
          }
        }
      }

      std::sort(neighborhood.begin(), neighborhood.end());

      for (int c = 0; c < channels; ++c) {
        float median = neighborhood[(neighborhood.size() / 2)];
        GetXYData(filtered_image, x, y)[c] = median;
      }
    }
  }

  image_ = filtered_image;
}

void FloatImage::Integrate(FloatImage* integral) const {
  integral->ResizeRowsCols(Rows() + 1, Cols() + 1);
  for (int i = 0; i < Channels(); i++) {
    // Fill the first row with zeros.
    for (int x = 0; x < Width(); x++) {
      integral->SetXY(x, 0, i, 0);
    }
    for (int y = 1; y <= Height(); y++) {
      // This variable is to correct floating point round off.
      float sum = 0;
      integral->SetXY(0, y, i, 0);
      for (int x = 1; x <= Width(); x++) {
        sum += this->GetXY(x - 1, y - 1, i);
        integral->SetXY(x, y, i, integral->GetXY(x, y - 1, i) + sum);
      }
    }
  }
}

void FloatImage::Resize(int new_width, int new_height, int num_channels) {
  const int old_width = Width();
  const int old_height = Height();
  const int channels = Channels();

  fipImage resized_image(
      image_.getImageType(), new_width, new_height, image_.getBitsPerPixel());

  float scale_x = static_cast<float>(old_width) / new_width;
  float scale_y = static_cast<float>(old_height) / new_height;

  for (int y = 0; y < new_height; ++y) {
    for (int x = 0; x < new_width; ++x) {
      float orig_x = x * scale_x;
      float orig_y = y * scale_y;

      int x0 = static_cast<int>(orig_x);
      int y0 = static_cast<int>(orig_y);
      int x1 = std::min(x0 + 1, old_width - 1);
      int y1 = std::min(y0 + 1, old_height - 1);

      float dx = orig_x - x0;
      float dy = orig_y - y0;

      const float* pixel_00 = GetXYData(image_, x0, y0);
      const float* pixel_01 = GetXYData(image_, x0, y1);
      const float* pixel_10 = GetXYData(image_, x1, y0);
      const float* pixel_11 = GetXYData(image_, x1, y1);

      for (int c = 0; c < channels; ++c) {
        float top_left = pixel_00[c];
        float top_right = pixel_10[c];
        float bottom_left = pixel_01[c];
        float bottom_right = pixel_11[c];

        float top_interp = top_left + dx * (top_right - top_left);
        float bottom_interp = bottom_left + dx * (bottom_right - bottom_left);

        float final_value = top_interp + dy * (bottom_interp - top_interp);
        GetXYData(resized_image, x, y)[c] = final_value;
      }
    }
  }

  image_ = resized_image;
}

void FloatImage::Resize(int new_width, int new_height) {
  Resize(new_width, new_height, Channels());
}

void FloatImage::ResizeRowsCols(int new_rows, int new_cols) {
  Resize(new_cols, new_rows, Channels());
}

void FloatImage::Resize(double scale) {
  Resize(static_cast<int>(scale * Width()),
         static_cast<int>(scale * Height()),
         Channels());
}

}  // namespace theia
