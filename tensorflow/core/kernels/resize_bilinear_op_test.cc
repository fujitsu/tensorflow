/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

#include <random>

namespace tensorflow {
enum class TestDevice { CPU, GPU };

class ResizeBilinearOpTestBase
    : public OpsTestBase,
      public ::testing::WithParamInterface<TestDevice> {
 protected:
  explicit ResizeBilinearOpTestBase()
      : align_corners_(false), half_pixel_centers_(false) {}

  void SetUp() override {
    if (GetParam() == TestDevice::GPU) {
      std::unique_ptr<Device> device_gpu(
          DeviceFactory::NewDevice("GPU", {}, "/job:a/replica:0/task:0"));
      SetDevice(DEVICE_GPU, std::move(device_gpu));
    }

    TF_EXPECT_OK(NodeDefBuilder("resize_bilinear_op", "ResizeBilinear")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", align_corners_)
                     .Attr("half_pixel_centers", half_pixel_centers_)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  const Tensor* SetRandomImageInput(const TensorShape& shape) {
    inputs_.clear();

    CHECK_EQ(shape.dims(), 4) << "All images must have 4 dimensions.";
    bool is_ref = IsRefType(input_types_[inputs_.size()]);
    Tensor* input = new Tensor(allocator(), DataTypeToEnum<float>::v(), shape);
    input->flat<float>().setRandom();
    tensors_.push_back(input);
    if (is_ref) {
      CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]),
               DataTypeToEnum<float>::v());
      inputs_.push_back({&lock_for_refs_, input});
    } else {
      CHECK_EQ(input_types_[inputs_.size()], DataTypeToEnum<float>::v());
      inputs_.push_back({nullptr, input});
    }
    return input;
  }

  // This is the straight forward unoptimized implementation of resize bilinear
  // We use this to confirm that the optimized version is exactly identical.
  void ResizeBilinearBaseline(TTypes<float, 4>::ConstTensor images,
                              TTypes<float, 4>::Tensor output) {
    const int batch = images.dimension(0);
    const int64 in_height = images.dimension(1);
    const int64 in_width = images.dimension(2);
    const int channels = images.dimension(3);

    ASSERT_EQ(batch, output.dimension(0));
    ASSERT_EQ(channels, output.dimension(3));

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);

    for (int b = 0; b < batch; ++b) {
      for (int64 y = 0; y < out_height; ++y) {
        const float in_y =
            half_pixel_centers_
                ? (static_cast<float>(y) + 0.5f) * height_scale - 0.5f
                : y * height_scale;
        const int64 top_y_index =
            std::max(static_cast<int64>(floorf(in_y)), static_cast<int64>(0));
        const int64 bottom_y_index =
            std::min(static_cast<int64>(ceilf(in_y)), in_height - 1);
        const float y_lerp = in_y - std::floor(in_y);
        for (int64 x = 0; x < out_width; ++x) {
          const float in_x =
              half_pixel_centers_
                  ? (static_cast<float>(x) + 0.5f) * width_scale - 0.5f
                  : x * width_scale;
          const int64 left_x_index =
              std::max(static_cast<int64>(floorf(in_x)), static_cast<int64>(0));
          const int64 right_x_index =
              std::min(static_cast<int64>(ceilf(in_x)), in_width - 1);
          const float x_lerp = in_x - std::floor(in_x);
          for (int c = 0; c < channels; ++c) {
            const float top_left = images(b, top_y_index, left_x_index, c);
            const float top_right = images(b, top_y_index, right_x_index, c);
            const float bottom_left =
                images(b, bottom_y_index, left_x_index, c);
            const float bottom_right =
                images(b, bottom_y_index, right_x_index, c);
            const float top = top_left + (top_right - top_left) * x_lerp;
            const float bottom =
                bottom_left + (bottom_right - bottom_left) * x_lerp;
            output(b, y, x, c) = top + (bottom - top) * y_lerp;
          }
        }
      }
    }
  }

  void TestResize(int batch_size, int input_width, int input_height,
                  int channels, int output_width, int output_height) {
    const TensorShape shape({batch_size, input_width, input_height, channels});
    const Tensor* input = SetRandomImageInput(shape);
    AddInputFromArray<int32>(TensorShape({2}), {output_width, output_height});
    TF_ASSERT_OK(RunOpKernel());

    std::unique_ptr<Tensor> expected(new Tensor(
        allocator(), DataTypeToEnum<float>::v(),
        TensorShape({batch_size, output_width, output_height, channels})));
    ResizeBilinearBaseline(input->tensor<float, 4>(),
                           expected->tensor<float, 4>());
    test::ExpectClose(*expected, *GetOutput(0), /*atol=*/3e-5);
  }

  void RunManyRandomTests(int channels) {
    for (int batch_size : {1, 2, 5}) {
      for (int in_w : {2, 4, 7, 20, 165}) {
        for (int in_h : {1, 3, 5, 8, 100, 233}) {
          for (int target_height : {1, 2, 3, 50, 113}) {
            for (int target_width : {target_height, target_height / 2 + 1}) {
              TestResize(batch_size, in_w, in_h, channels, target_width,
                         target_height);
            }
          }
        }
      }
    }
  }

  bool align_corners_;
  bool half_pixel_centers_;
};

class ResizeBilinearOpTest : public ResizeBilinearOpTestBase {
 public:
  ResizeBilinearOpTest() {}
};

class ResizeBilinearHalfPixelCentersOpTest : public ResizeBilinearOpTestBase {
 public:
  ResizeBilinearHalfPixelCentersOpTest() { half_pixel_centers_ = true; }
};

class ResizeBilinearOpAlignCornersTest : public ResizeBilinearOpTestBase {
 public:
  ResizeBilinearOpAlignCornersTest() { align_corners_ = true; }
};

TEST_P(ResizeBilinearOpTest, TestResizeRandomDataSeveralInputsSizes1Channel) {
  RunManyRandomTests(1);
}

TEST_P(ResizeBilinearOpTest, TestResizeRandomDataSeveralInputsSizes3Channels) {
  RunManyRandomTests(3);
}

TEST_P(ResizeBilinearOpTest, TestResizeRandomDataSeveralInputsSizes4Channels) {
  RunManyRandomTests(4);
}

TEST_P(ResizeBilinearOpTest, TestBilinear2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {1.0});
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinearRandom2x2To1x1) {
  const Tensor* input = SetRandomImageInput(TensorShape({1, 2, 2, 1}));
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor* output = GetOutput(0);
  std::unique_ptr<Tensor> expected(new Tensor(
      allocator(), DataTypeToEnum<float>::v(), TensorShape({1, 1, 1, 1})));
  ResizeBilinearBaseline(input->tensor<float, 4>(),
                         expected->tensor<float, 4>());
  EXPECT_EQ(input->flat<float>()(0), output->flat<float>()(0));
  test::ExpectClose(*expected, *output);
}

TEST_P(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {1.0});
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,        5.0f / 3,  2,
     7.0f / 3, 3,         10.0f / 3,
     3,        11.0f / 3, 4});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // The corners exactly align with the original corners, and we bilinear
  // interpolate the values in between.

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  1.5,  2,
     2,  2.5,  3,
     3,  3.5,  4});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear3x3To2x2) {
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,   2.5,
     5.5,   7});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners3x3To2x2) {
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  3,
     7,  9});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear3x3To4x4) {
  // Input:
  //  1, 2, 3,
  //  4, 5, 6,
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1.75, 2.5, 3,
     3.25, 4, 4.75, 5.25,
     5.5, 6.25, 7, 7.5,
     7,  7.75, 8.5, 9});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear4x4To3x3) {
  // Input:
  //  1,  2,  3,  4
  //  5,  6,  7,  8
  //  9, 10, 11, 12
  // 13, 14, 15, 16
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,        7.0f/3, 11.0f/3,
     19.0f/3, 23.0f/3, 27.0f/3,
     35.0f/3, 39.0f/3, 43.0f/3});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearHalfPixelCentersOpTest, TestDownsamples) {
  TestResize(4, 298, 297, 3, 61, 71);
}

TEST_P(ResizeBilinearHalfPixelCentersOpTest, TestUpsamples) {
  TestResize(4, 61, 71, 3, 298, 297);
}

TEST_P(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners4x4To3x3) {
  // Input:
  //  1,  2,  3,  4
  //  5,  6,  7,  8
  //  9, 10, 11, 12
  // 13, 14, 15, 16
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    { 1,  2.5,  4,
      7,  8.5, 10,
     13, 14.5, 16});

  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear2x2To3x3Batch2) {
  // Input:
  //  1, 2
  //  3, 4
  //
  // repeated twice
  AddInputFromArray<float>(TensorShape({2, 2, 2, 1}), {1, 2, 3, 4, 1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1, 5.0f/3, 2, 7.0f/3, 3, 10.0f/3, 3, 11.0f/3, 4,
     1, 5.0f/3, 2, 7.0f/3, 3, 10.0f/3, 3, 11.0f/3, 4
    });
  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear2x2x2To3x3x2) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 2}),
                           {1, -1, 2, -2, 3, -3, 4, -4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 2}));
  // clang-format off
  test::FillValues<float>(&expected,
    {
      1,       -1,
      5.0f/3,  -5.0f/3,
      2,       -2,
      7.0f/3,  -7.0f/3,
      3,       -3,
      10.0f/3, -10.0f/3,
      3,       -3,
      11.0f/3, -11.0f/3,
      4,       -4
    });
  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

TEST_P(ResizeBilinearOpTest, TestBilinear2x2To4x4) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1,  1.5, 2, 2,
     2,  2.5, 3, 3,
     3,  3.5, 4, 4,
     3,  3.5, 4, 4});
  // clang-format on
  test::ExpectClose(expected, *GetOutput(0));
}

// similar_size case
TEST_P(ResizeBilinearOpTest, Test1_1c) { TestResize(1, 183, 299, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test1_3c) { TestResize(1, 183, 299, 3, 299, 299); }

// Significantly smaller: scale_up case
TEST_P(ResizeBilinearOpTest, Test2_1c) { TestResize(1, 141, 186, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test2_3c) { TestResize(1, 141, 186, 3, 299, 299); }

// Significantly larger: scale_down case
TEST_P(ResizeBilinearOpTest, Test3_1c) { TestResize(1, 749, 603, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test3_3c) { TestResize(1, 749, 603, 3, 299, 299); }

// Exactly the same size
TEST_P(ResizeBilinearOpTest, Test4_1c) { TestResize(1, 299, 299, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test4_3c) { TestResize(1, 299, 299, 3, 299, 299); }

// Slightly smaller: similar_size case
TEST_P(ResizeBilinearOpTest, Test5_1c) { TestResize(1, 298, 297, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test5_3c) { TestResize(1, 298, 297, 3, 299, 299); }

// Slightly bigger: similar_size case
TEST_P(ResizeBilinearOpTest, Test6_1c) { TestResize(1, 304, 303, 1, 299, 299); }
TEST_P(ResizeBilinearOpTest, Test6_3c) { TestResize(1, 304, 303, 3, 299, 299); }

TEST_P(ResizeBilinearOpTest, TestInvalidOutputSize) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "Invalid argument: output dimensions must be positive"))
      << s;
}

TEST_P(ResizeBilinearOpTest, TestInvalidInputShape) {
  AddInputFromArray<float>(TensorShape({2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "Invalid argument: input must be 4-dimensional"))
      << s;
}

TEST_P(ResizeBilinearOpTest, TestInvalidSizeDim) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {4, 4});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "Invalid argument: shape_t must be 1-dimensional"))
      << s;
}

TEST_P(ResizeBilinearOpTest, TestInvalidSizeElements) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {4, 4, 1});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "Invalid argument: shape_t must have two elements"))
      << s;
}

INSTANTIATE_TEST_SUITE_P(ResizeBilinearOpTestCpu, ResizeBilinearOpTest,
                         ::testing::Values(TestDevice::CPU));
INSTANTIATE_TEST_SUITE_P(ResizeBilinearHalfPixelCentersOpTestCpu,
                         ResizeBilinearHalfPixelCentersOpTest,
                         ::testing::Values(TestDevice::CPU));
INSTANTIATE_TEST_SUITE_P(ResizeBilinearOpAlignCornersTestCpu,
                         ResizeBilinearOpAlignCornersTest,
                         ::testing::Values(TestDevice::CPU));
#if GOOGLE_CUDA
// Instantiate tests for GPU.
INSTANTIATE_TEST_SUITE_P(ResizeBilinearOpTestGpu, ResizeBilinearOpTest,
                         ::testing::Values(TestDevice::GPU));
INSTANTIATE_TEST_SUITE_P(ResizeBilinearHalfPixelCentersOpTestGpu,
                         ResizeBilinearHalfPixelCentersOpTest,
                         ::testing::Values(TestDevice::GPU));
INSTANTIATE_TEST_SUITE_P(ResizeBilinearOpAlignCornersTestGpu,
                         ResizeBilinearOpAlignCornersTest,
                         ::testing::Values(TestDevice::GPU));
#endif  // GOOGLE_CUDA

class ResizeBilinearGradOpTestBase
    : public OpsTestBase,
      public ::testing::WithParamInterface<TestDevice> {
 protected:
  explicit ResizeBilinearGradOpTestBase()
      : align_corners_(false), half_pixel_centers_(false) {}

  void MakeOp(bool align_corners=false, bool half_pixel_centers=false) {
    align_corners_ = align_corners;
    half_pixel_centers_ = half_pixel_centers;

    TF_EXPECT_OK(NodeDefBuilder("resize_bilinear_grad_op", "ResizeBilinearGrad")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("align_corners", align_corners)
                     .Attr("half_pixel_centers", half_pixel_centers)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  // CalculateResizeScale determines the float scaling factor.
  inline float CalculateResizeScale(int64 in_size, int64 out_size) {
    return (align_corners_ && out_size > 1)
              ? (in_size - 1) / static_cast<float>(out_size - 1)
              : in_size / static_cast<float>(out_size);
  }

  // Half pixel scaler scales assuming that the pixel centers are at 0.5, i.e. the
  // floating point coordinates of the top,left pixel is 0.5,0.5.
  inline float HalfPixelScaler(const int x, const float scale) const {
      // Note that we subtract 0.5 from the return value, as the existing bilinear
      // sampling code etc assumes pixels are in the old coordinate system.
      return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
  }

  // Older incorrect scaling method that causes all resizes to have a slight
  // translation leading to inconsistent results. For example, a flip then a
  // resize gives different results then a resize then a flip.
  inline float LegacyScaler(const int x, const float scale) const {
      return static_cast<float>(x) * scale;
  }

  void ResizeGradCoreSingle(typename TTypes<float, 4>::ConstTensor input_grad,
                            const float height_scale, const float width_scale,
                            typename TTypes<float, 4>::Tensor output_grad) {
    const Eigen::Index batch = output_grad.dimension(0);
    const Eigen::Index original_height = output_grad.dimension(1);
    const Eigen::Index original_width = output_grad.dimension(2);
    const Eigen::Index channels = output_grad.dimension(3);

    const Eigen::Index resized_height = input_grad.dimension(1);
    const Eigen::Index resized_width = input_grad.dimension(2);

    output_grad.setZero();

    // Each resized output pixel was computed as a weighted average of four
    // input pixels. Here we find the four input pixel locations that
    // contributed to each output pixel and propagate the gradient at the output
    // pixel location to each of those four input pixel locations in the same
    // proportions that they originally contributed to the output pixel.
    // Here is the forward-propagation pseudo-code, for reference:
    // resized(b, y, x, c) = top_left     * (1 - y) * (1 - x)
    //                     + top_right    * (1 - y) *      x
    //                     + bottom_left  *      y  * (1 - x)
    //                     + bottom_right *      y  *      x
    for (Eigen::Index b = 0; b < batch; ++b) {
      for (Eigen::Index y = 0; y < resized_height; ++y) {
        const float in_y = half_pixel_centers_ ? HalfPixelScaler(y, height_scale)
                                               : LegacyScaler(y, height_scale);
        const Eigen::Index top_y_index =
            std::max(static_cast<Eigen::Index>(floorf(in_y)),
                     static_cast<Eigen::Index>(0));
        const Eigen::Index bottom_y_index = std::min(
            static_cast<Eigen::Index>(ceilf(in_y)), original_height - 1);
        const float y_lerp = in_y - floorf(in_y);
        const float inverse_y_lerp = (1.0f - y_lerp);
        for (Eigen::Index x = 0; x < resized_width; ++x) {
          const float in_x = half_pixel_centers_ ? HalfPixelScaler(x, width_scale)
                                                 : LegacyScaler(x, width_scale);
          const Eigen::Index left_x_index =
              std::max(static_cast<Eigen::Index>(floorf(in_x)),
                       static_cast<Eigen::Index>(0));
          const Eigen::Index right_x_index = std::min(
              static_cast<Eigen::Index>(ceilf(in_x)), original_width - 1);
          const float x_lerp = in_x - floorf(in_x);
          const float inverse_x_lerp = (1.0f - x_lerp);
          for (Eigen::Index c = 0; c < channels; ++c) {
            output_grad(b, top_y_index, left_x_index, c) +=
                float(input_grad(b, y, x, c) * inverse_y_lerp * inverse_x_lerp);
            output_grad(b, top_y_index, right_x_index, c) +=
                float(input_grad(b, y, x, c) * inverse_y_lerp * x_lerp);
            output_grad(b, bottom_y_index, left_x_index, c) +=
                float(input_grad(b, y, x, c) * y_lerp * inverse_x_lerp);
            output_grad(b, bottom_y_index, right_x_index, c) +=
                float(input_grad(b, y, x, c) * y_lerp * x_lerp);
          }
        }
      }
    }
  }

  void RunFloatTest(bool align_corners, bool half_pixel_centers,
                    int batch, int in_height, int in_width,
                    int out_height,int out_width,  int channels) {

    MakeOp(align_corners, half_pixel_centers);

    // set input data 1
    AddInput<float>(TensorShape({batch, in_height, in_width, channels}),
                    [](int x) {return float(rand())/(RAND_MAX/10) - 5.0;});

    // set input data 2
    AddInput<float>(TensorShape({batch, out_height, out_width, channels}),
                    [](int x) {return float(rand())/(RAND_MAX/10) - 5.0;});

    // run OpKernel
    TF_ASSERT_OK(RunOpKernel());

    // make expected data(by single process)
    Tensor expected(allocator(), DT_FLOAT, TensorShape({batch, out_height, out_width, channels}));

    TTypes<float, 4>::ConstTensor input_grad = GetInput(0).tensor<float, 4>();
    typename TTypes<float, 4>::Tensor output_grad(expected.tensor<float, 4>());

    int64 resized_height = GetInput(0).dim_size(1);
    int64 resized_width = GetInput(0).dim_size(2);
    int64 original_height = GetInput(1).dim_size(1);
    int64 original_width = GetInput(1).dim_size(2);

    float height_scale = CalculateResizeScale(original_height, resized_height);
    float width_scale = CalculateResizeScale(original_width, resized_width);

    ResizeGradCoreSingle(input_grad, height_scale, width_scale, output_grad);

    // prepare value
    test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);
  }

  bool align_corners_;
  bool half_pixel_centers_;
};    // ResizeBilinearGradOpTestBase

class ResizeBilinearGradOpTest : public ResizeBilinearGradOpTestBase {
 public:
  ResizeBilinearGradOpTest() {}
};
TEST_F(ResizeBilinearGradOpTest, test_1) {
  RunFloatTest(false, false, 12, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_2) {
  RunFloatTest(false, false, 12, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_3) {
  RunFloatTest(false, false, 12, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_4) {
  RunFloatTest(false, false, 12, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_5) {
  RunFloatTest(false, false, 12, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_6) {
  RunFloatTest(false, false, 12, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_7) {
  RunFloatTest(false, false, 24, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_8) {
  RunFloatTest(false, false, 24, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_9) {
  RunFloatTest(false, false, 24, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_10) {
  RunFloatTest(false, false, 24, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_11) {
  RunFloatTest(false, false, 24, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_12) {
  RunFloatTest(false, false, 24, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_13) {
  RunFloatTest(false, false, 48, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_14) {
  RunFloatTest(false, false, 48, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_15) {
  RunFloatTest(false, false, 48, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_16) {
  RunFloatTest(false, false, 48, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_17) {
  RunFloatTest(false, false, 48, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_18) {
  RunFloatTest(false, false, 48, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_19) {
  RunFloatTest(false, false, 64, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_20) {
  RunFloatTest(false, false, 64, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_21) {
  RunFloatTest(false, false, 64, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_22) {
  RunFloatTest(false, false, 64, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_23) {
  RunFloatTest(false, false, 64, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_24) {
  RunFloatTest(false, false, 64, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_25) {
  RunFloatTest(false, false, 128, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_26) {
  RunFloatTest(false, false, 128, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_27) {
  RunFloatTest(false, false, 128, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_28) {
  RunFloatTest(false, false, 128, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_29) {
  RunFloatTest(false, false, 128, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_30) {
  RunFloatTest(false, false, 128, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_31) {
  RunFloatTest(false, false, 256, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_32) {
  RunFloatTest(false, false, 256, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_33) {
  RunFloatTest(false, false, 256, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_34) {
  RunFloatTest(false, false, 256, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_35) {
  RunFloatTest(false, false, 256, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_36) {
  RunFloatTest(false, false, 256, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_37) {
  RunFloatTest(false, true, 12, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_38) {
  RunFloatTest(false, true, 12, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_39) {
  RunFloatTest(false, true, 12, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_40) {
  RunFloatTest(false, true, 12, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_41) {
  RunFloatTest(false, true, 12, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_42) {
  RunFloatTest(false, true, 12, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_43) {
  RunFloatTest(false, true, 24, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_44) {
  RunFloatTest(false, true, 24, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_45) {
  RunFloatTest(false, true, 24, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_46) {
  RunFloatTest(false, true, 24, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_47) {
  RunFloatTest(false, true, 24, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_48) {
  RunFloatTest(false, true, 24, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_49) {
  RunFloatTest(false, true, 48, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_50) {
  RunFloatTest(false, true, 48, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_51) {
  RunFloatTest(false, true, 48, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_52) {
  RunFloatTest(false, true, 48, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_53) {
  RunFloatTest(false, true, 48, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_54) {
  RunFloatTest(false, true, 48, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_55) {
  RunFloatTest(false, true, 64, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_56) {
  RunFloatTest(false, true, 64, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_57) {
  RunFloatTest(false, true, 64, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_58) {
  RunFloatTest(false, true, 64, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_59) {
  RunFloatTest(false, true, 64, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_60) {
  RunFloatTest(false, true, 64, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_61) {
  RunFloatTest(false, true, 128, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_62) {
  RunFloatTest(false, true, 128, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_63) {
  RunFloatTest(false, true, 128, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_64) {
  RunFloatTest(false, true, 128, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_65) {
  RunFloatTest(false, true, 128, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_66) {
  RunFloatTest(false, true, 128, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_67) {
  RunFloatTest(false, true, 256, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_68) {
  RunFloatTest(false, true, 256, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_69) {
  RunFloatTest(false, true, 256, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_70) {
  RunFloatTest(false, true, 256, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_71) {
  RunFloatTest(false, true, 256, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_72) {
  RunFloatTest(false, true, 256, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_73) {
  RunFloatTest(true, false, 12, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_74) {
  RunFloatTest(true, false, 12, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_75) {
  RunFloatTest(true, false, 12, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_76) {
  RunFloatTest(true, false, 12, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_77) {
  RunFloatTest(true, false, 12, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_78) {
  RunFloatTest(true, false, 12, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_79) {
  RunFloatTest(true, false, 24, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_80) {
  RunFloatTest(true, false, 24, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_81) {
  RunFloatTest(true, false, 24, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_82) {
  RunFloatTest(true, false, 24, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_83) {
  RunFloatTest(true, false, 24, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_84) {
  RunFloatTest(true, false, 24, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_85) {
  RunFloatTest(true, false, 48, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_86) {
  RunFloatTest(true, false, 48, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_87) {
  RunFloatTest(true, false, 48, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_88) {
  RunFloatTest(true, false, 48, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_89) {
  RunFloatTest(true, false, 48, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_90) {
  RunFloatTest(true, false, 48, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_91) {
  RunFloatTest(true, false, 64, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_92) {
  RunFloatTest(true, false, 64, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_93) {
  RunFloatTest(true, false, 64, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_94) {
  RunFloatTest(true, false, 64, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_95) {
  RunFloatTest(true, false, 64, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_96) {
  RunFloatTest(true, false, 64, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_97) {
  RunFloatTest(true, false, 128, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_98) {
  RunFloatTest(true, false, 128, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_99) {
  RunFloatTest(true, false, 128, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_100) {
  RunFloatTest(true, false, 128, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_101) {
  RunFloatTest(true, false, 128, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_102) {
  RunFloatTest(true, false, 128, 3, 3, 9, 9, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_103) {
  RunFloatTest(true, false, 256, 64, 64, 32, 32, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_104) {
  RunFloatTest(true, false, 256, 64, 64, 32, 32, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_105) {
  RunFloatTest(true, false, 256, 33, 33, 1, 1, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_106) {
  RunFloatTest(true, false, 256, 33, 33, 1, 1, 1024);
}

TEST_F(ResizeBilinearGradOpTest, test_107) {
  RunFloatTest(true, false, 256, 3, 3, 9, 9, 128);
}

TEST_F(ResizeBilinearGradOpTest, test_108) {
  RunFloatTest(true, false, 256, 3, 3, 9, 9, 1024);
}

}  // namespace tensorflow
