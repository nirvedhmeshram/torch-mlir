// RUN: npcomp-opt -torch-refine-types -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[ARG:.*]]: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
// CHECK:           %[[SHAPED:.*]] = numpy.tensor_static_info_cast %[[ARG]] : tensor<2x3x?xf32> to tensor<2x3x?xf32>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.tensor_static_info_cast %[[SHAPED]] : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : tensor<*x!numpy.any_dtype>
func @f(%arg0: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %0 = numpy.tensor_static_info_cast %arg0 : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
  return %0 : tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[ARG:.*]]: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
// CHECK:           %[[SHAPED:.*]] = torch.aten.tanh %[[ARG]] : tensor<2x3x?xf32> -> tensor<2x3x?xf32>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.tensor_static_info_cast %[[SHAPED]] : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : tensor<*x!numpy.any_dtype>
func @f(%arg0: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %1 = torch.aten.tanh %arg0 : tensor<2x3x?xf32> -> tensor<*x!numpy.any_dtype>
  return %1 : tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[LHS:.*]]: tensor<2x?xf32>,
// CHECK-SAME:            %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<*x!numpy.any_dtype> {
// CHECK:           %[[MM:.*]] = torch.aten.mm %[[LHS]], %[[RHS]] : tensor<2x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.tensor_static_info_cast %[[MM]] : tensor<?x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : tensor<*x!numpy.any_dtype>
func @f(%arg0: tensor<2x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %1 = torch.aten.mm %arg0, %arg1 : tensor<2x?xf32>, tensor<?x?xf32> -> tensor<*x!numpy.any_dtype>
  return %1 : tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[INPUT:.*]]: tensor<?x3xf32>,
// CHECK-SAME:            %[[WEIGHT:.*]]: tensor<5x3xf32>,
// CHECK-SAME:            %[[BIAS:.*]]: tensor<5xf32>) -> tensor<*x!numpy.any_dtype> {
// CHECK:           %[[LINEAR:.*]] = torch.aten.linear %[[INPUT]], %[[WEIGHT]], %[[BIAS]] : tensor<?x3xf32>, tensor<5x3xf32>, tensor<5xf32> -> tensor<?x?xf32>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.tensor_static_info_cast %[[LINEAR]] : tensor<?x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : tensor<*x!numpy.any_dtype>
func @f(%arg0: tensor<?x3xf32>, %arg1: tensor<5x3xf32>, %arg2: tensor<5xf32>) -> tensor<*x!numpy.any_dtype> {
  %1 = torch.aten.linear %arg0, %arg1, %arg2 : tensor<?x3xf32>, tensor<5x3xf32>, tensor<5xf32> -> tensor<*x!numpy.any_dtype>
  return %1 : tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL: func @f
// CHECK:           %[[CONV2D:.*]] = torch.aten.conv2d{{.*}} -> tensor<?x?x?x?x!numpy.any_dtype>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.tensor_static_info_cast %[[CONV2D]] : tensor<?x?x?x?x!numpy.any_dtype> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : tensor<*x!numpy.any_dtype>
func @f(%arg0:tensor<*x!numpy.any_dtype>, %arg1:tensor<*x!numpy.any_dtype>, %arg2:tensor<*x!numpy.any_dtype>) ->tensor<*x!numpy.any_dtype> {
  %c0_i64 = constant 0 : i64
  %c1_i64 = constant 1 : i64
  %0 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %1 = basicpy.build_list %c0_i64, %c0_i64 : (i64, i64) -> !basicpy.ListType
  %2 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %3 = torch.aten.conv2d %arg0, %arg1, %arg2, %0, %1, %2, %c1_i64 : tensor<*x!numpy.any_dtype>, tensor<*x!numpy.any_dtype>, tensor<*x!numpy.any_dtype>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i64 ->tensor<*x!numpy.any_dtype>
  return %3 :tensor<*x!numpy.any_dtype>
}

// CHECK-LABEL: func @g
// CHECK:           %[[CONV2D:.*]] = torch.aten.conv2d{{.*}} -> tensor<?x?x?x?xf32>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.tensor_static_info_cast %[[CONV2D]] : tensor<?x?x?x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : tensor<*x!numpy.any_dtype>
func @g(%arg0:tensor<*xf32>, %arg1:tensor<*xf32>, %arg2:tensor<*xf32>) ->tensor<*x!numpy.any_dtype> {
  %c0_i64 = constant 0 : i64
  %c1_i64 = constant 1 : i64
  %0 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %1 = basicpy.build_list %c0_i64, %c0_i64 : (i64, i64) -> !basicpy.ListType
  %2 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %3 = torch.aten.conv2d %arg0, %arg1, %arg2, %0, %1, %2, %c1_i64 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i64 ->tensor<*x!numpy.any_dtype>
  return %3 :tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: tensor<?x?x?x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %c1_i64 = constant 1 : i64
  %c3_i64 = constant 3 : i64
  %c2_i64 = constant 2 : i64
  %bool_false = basicpy.bool_constant false
  %21 = basicpy.build_list %c3_i64, %c3_i64 : (i64, i64) -> !basicpy.ListType
  %22 = basicpy.build_list %c2_i64, %c2_i64 : (i64, i64) -> !basicpy.ListType
  %23 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %24 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  // CHECK: torch.aten.max_pool2d{{.*}} -> tensor<?x?x?x?xf32>
  %27 = torch.aten.max_pool2d %arg0, %21, %22, %23, %24, %bool_false : tensor<?x?x?x?xf32>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, !basicpy.BoolType -> tensor<*x!numpy.any_dtype>
  return %27 : tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: tensor<?x?x?x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %c1_i64 = constant 1 : i64
  %0 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  // CHECK: torch.aten.adaptive_avg_pool2d{{.*}} -> tensor<?x?x?x?xf32>
  %1 = torch.aten.adaptive_avg_pool2d %arg0, %0 : tensor<?x?x?x?xf32>, !basicpy.ListType -> tensor<*x!numpy.any_dtype>
  return %1 : tensor<*x!numpy.any_dtype>
}

// -----

// Also test cast insertion for array types.
// CHECK-LABEL:   func @flatten_all(
// CHECK:           %[[FLATTENED:.*]] = torch.aten.flatten.using_ints{{.*}}-> !numpy.ndarray<[?]:f32>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.static_info_cast %[[FLATTENED]] : !numpy.ndarray<[?]:f32> to !numpy.ndarray<*:!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]]
func @flatten_all(%arg0: !numpy.ndarray<[3,2,?,5]:f32>) -> !numpy.ndarray<*:!numpy.any_dtype> {
  %end = constant -1 : i64
  %start = constant 0 : i64
  %0 = torch.aten.flatten.using_ints %arg0, %start, %end : !numpy.ndarray<[3,2,?,5]:f32>, i64, i64 -> !numpy.ndarray<*:!numpy.any_dtype>
  return %0 : !numpy.ndarray<*:!numpy.any_dtype>
}

// CHECK-LABEL:   func @flatten_some(
// CHECK:           torch.aten.flatten.using_ints{{.*}}-> !numpy.ndarray<[3,?,5]:f32>
func @flatten_some(%arg0: !numpy.ndarray<[3,2,?,5]:f32>) -> !numpy.ndarray<*:!numpy.any_dtype> {
  %end = constant -2 : i64
  %start = constant 1 : i64
  %0 = torch.aten.flatten.using_ints %arg0, %start, %end : !numpy.ndarray<[3,2,?,5]:f32>, i64, i64 -> !numpy.ndarray<*:!numpy.any_dtype>
  return %0 : !numpy.ndarray<*:!numpy.any_dtype>
}

// CHECK-LABEL:   func @flatten_rank0(
// CHECK:           torch.aten.flatten.using_ints{{.*}}-> !numpy.ndarray<[?]:f32>
func @flatten_rank0(%arg0: !numpy.ndarray<[]:f32>) -> !numpy.ndarray<*:!numpy.any_dtype> {
  %end = constant -1 : i64
  %start = constant 0 : i64
  %0 = torch.aten.flatten.using_ints %arg0, %start, %end : !numpy.ndarray<[]:f32>, i64, i64 -> !numpy.ndarray<*:!numpy.any_dtype>
  return %0 : !numpy.ndarray<*:!numpy.any_dtype>
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: tensor<4x6x3xf32>, %arg1: tensor<1x1x3xf32>, %arg2: tensor<?x3xf32>) {
  %c1_i64 = constant 1 : i64
  // CHECK: torch.aten.add{{.*}} -> tensor<?x?x?xf32>
  %0 = torch.aten.add.Tensor %arg0, %arg1, %c1_i64 : tensor<4x6x3xf32>, tensor<1x1x3xf32>, i64 -> tensor<*x!numpy.any_dtype>
  // CHECK: torch.aten.add{{.*}} -> tensor<?x?x?xf32>
  %1 = torch.aten.add.Tensor %arg0, %arg2, %c1_i64 : tensor<4x6x3xf32>, tensor<?x3xf32>, i64 -> tensor<*x!numpy.any_dtype>
  return
}

// -----

// CHECK-LABEL:   func @f
func @f(%arg0: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
  // Check propagation through multiple ops.
  // CHECK:           torch.aten.tanh %{{.*}} : tensor<2x3x?xf32> -> tensor<2x3x?xf32>
  // CHECK:           torch.aten.tanh %{{.*}} : tensor<2x3x?xf32> -> tensor<2x3x?xf32>
  // CHECK:           torch.aten.tanh %{{.*}} : tensor<2x3x?xf32> -> tensor<2x3x?xf32>
  %1 = torch.aten.tanh %arg0 : tensor<2x3x?xf32> -> tensor<*x!numpy.any_dtype>
  %2 = torch.aten.tanh %1 : tensor<*x!numpy.any_dtype> -> tensor<*x!numpy.any_dtype>
  %3 = torch.aten.tanh %2 : tensor<*x!numpy.any_dtype> -> tensor<*x!numpy.any_dtype>
  return %3 : tensor<*x!numpy.any_dtype>
}

// -----

// Check rewriting logic in case of mixes of users that do/don't allow type
// refinement.
// CHECK-LABEL:   func @f
func @f(%arg0: tensor<2x3x?xf32>) -> (tensor<*x!numpy.any_dtype>, tensor<*x!numpy.any_dtype>) {
  // CHECK: %[[REFINED_TYPE:.*]] = torch.aten.tanh %{{.*}} : tensor<2x3x?xf32> -> tensor<2x3x?xf32>
  %1 = torch.aten.tanh %arg0 : tensor<2x3x?xf32> -> tensor<*x!numpy.any_dtype>
  // CHECK: %[[ORIGINAL_TYPE:.*]] = numpy.tensor_static_info_cast %[[REFINED_TYPE]] : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
  // CHECK: torch.aten.tanh %[[REFINED_TYPE]] : tensor<2x3x?xf32> -> tensor<2x3x?xf32>
  %3 = torch.aten.tanh %1 : tensor<*x!numpy.any_dtype> -> tensor<*x!numpy.any_dtype>
  // CHECK: return %[[ORIGINAL_TYPE]], %[[ORIGINAL_TYPE]] : tensor<*x!numpy.any_dtype>, tensor<*x!numpy.any_dtype>
  return %1, %1 : tensor<*x!numpy.any_dtype>, tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL:   func @f
// CHECK: %[[ATEN:.*]] = torch.aten.tanh %{{.*}} : tensor<*x!numpy.any_dtype> -> tensor<2x3x?xf32>
// CHECK: %[[CAST:.*]] = numpy.tensor_static_info_cast %[[ATEN]] : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK: return %[[CAST]] : tensor<*x!numpy.any_dtype>
func @f(%arg0: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %cast = numpy.tensor_static_info_cast %arg0 : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
  br ^bb1(%cast: tensor<*x!numpy.any_dtype>)
^bb1(%arg1: tensor<*x!numpy.any_dtype>):
  %1 = torch.aten.tanh %arg1 : tensor<*x!numpy.any_dtype> -> tensor<*x!numpy.any_dtype>
  return %1 : tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL:   func @f
// CHECK: func private @callee
// CHECK-NEXT: torch.aten.tanh %{{.*}} : tensor<*x!numpy.any_dtype> -> tensor<2x3x?xf32>
func @f() {
  module {
    func private @callee(%arg0: tensor<*x!numpy.any_dtype>) {
      %1 = torch.aten.tanh %arg0 : tensor<*x!numpy.any_dtype> -> tensor<*x!numpy.any_dtype>
      return
    }
    func @caller(%arg0: tensor<2x3x?xf32>) {
      %cast = numpy.tensor_static_info_cast %arg0 : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
      call @callee(%cast) : (tensor<*x!numpy.any_dtype>) -> ()
      return
    }
  }
  return
}