load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "quantize_model",
    srcs = [
        "quantize_model.cc",
    ],
    hdrs = [
        "quantize_model.h",
    ],
    deps = [
        "//larq_compute_engine/mlir:larq_compute_engine_sanitize",
        "@flatbuffers",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:common",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_quantize",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util",
    ],
)

pybind_extension(
    name = "_calibration_wrapper",
    srcs = ["calibration_wrapper.cc"],
    module_name = "calibration_wrapper",
    deps = [
        ":quantize_model",
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/python/interpreter_wrapper:numpy",
        "@org_tensorflow//tensorflow/lite/python/interpreter_wrapper:python_error_reporter",
        "@org_tensorflow//tensorflow/lite/tools/optimize/calibration:calibration_reader",
        "@org_tensorflow//tensorflow/lite/tools/optimize/calibration:calibrator_lib",
        "@pybind11",
    ],
)
