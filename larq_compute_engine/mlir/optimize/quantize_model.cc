#include "quantize_model.h"

#include "absl/strings/string_view.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "larq_compute_engine/mlir/transforms/passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/QuantOps/QuantOps.h"  // TF:llvm-project
#include "mlir/IR/Function.h"                // TF:llvm-project
#include "mlir/IR/Location.h"                // TF:llvm-project
#include "mlir/IR/MLIRContext.h"             // TF:llvm-project
#include "mlir/IR/Module.h"                  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"         // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_translate.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir {
namespace lite {

struct SanitizeLCE : public RewritePattern {
  SanitizeLCE(MLIRContext* context)
      : RewritePattern("tfl.UNSUPPORTED_custom_LceBconv2d", 1, context) {}

  PatternMatchResult match(Operation* op) const override {
    // No need for further matches, we only care about the name
    return matchSuccess();
  }

  void rewrite(Operation* op, PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<TF::LceBconv2dOp>(
        op, op->getResultTypes(), op->getOperands(), op->getAttrs());
  }
};

struct RenameLCE : public FunctionPass<RenameLCE> {
  void runOnFunction() override {
    FuncOp func = getFunction();
    OwningRewritePatternList patterns;
    patterns.insert<SanitizeLCE>(func.getContext());
    applyPatternsGreedily(func, patterns);
  }
};

std::unique_ptr<OpPassBase<FuncOp>> CreateRenameLCEPass() {
  return std::make_unique<RenameLCE>();
}

// TODO(fengliuai): check the result for `allow_float` flag.
TfLiteStatus QuantizeModel(
    const tflite::ModelT& input_model, const tflite::TensorType& input_type,
    const tflite::TensorType& output_type,
    const std::unordered_set<std::string>& operator_names, bool allow_float,
    flatbuffers::FlatBufferBuilder* builder,
    tflite::ErrorReporter* error_reporter) {
  // TODO(b/142502494): remove this restriction by improving the `emit_adaptor`
  // flag
  if (input_type != output_type) {
    error_reporter->Report("Required same input type and output type.");
    return kTfLiteError;
  }

  MLIRContext context;
  StatusScopedDiagnosticHandler statusHandler(&context,
                                              /*propagate=*/true);

  // Import input_model to a MLIR module
  flatbuffers::FlatBufferBuilder input_builder;
  flatbuffers::Offset<tflite::Model> input_model_location =
      tflite::Model::Pack(input_builder, &input_model);
  tflite::FinishModelBuffer(input_builder, input_model_location);

  std::string serialized_model(
      reinterpret_cast<const char*>(input_builder.GetBufferPointer()),
      input_builder.GetSize());

  OwningModuleRef module = tflite::FlatBufferToMlir(serialized_model, &context,
                                                    UnknownLoc::get(&context));
  if (!module) {
    error_reporter->Report("Couldn't import flatbuffer to MLIR.");
    return kTfLiteError;
  }

  // Apply quantization passes
  PassManager pm(module->getContext());
  TFL::QuantizationSpecs quant_specs;
  quant_specs.inference_type = tensorflow::DT_QINT8;
  quant_specs.post_training_quantization = true;

  bool emit_adaptor = false;
  auto input_tf_type = tflite::TflTypeToTfType(input_type);
  if (input_tf_type == tensorflow::DT_FLOAT) {
    emit_adaptor = true;
  } else if (input_tf_type == tensorflow::DT_UINT8) {
    quant_specs.inference_type = tensorflow::DT_QUINT8;
  }

  pm.addPass(CreateRenameLCEPass());

  pm.addPass(TFL::CreatePrepareQuantizePass(quant_specs));
  pm.addPass(TFL::CreateQuantizePass());
  pm.addPass(TFL::CreatePostQuantizePass(emit_adaptor));

  // Clean up dangling LCE ops
  pm.addPass(mlir::TFL::CreateOptimizeLCEPass());

  // Cleaning up the LCE ops also causes dangling quantize/dequantize ops
  // so we clean those up again
  pm.addPass(TFL::CreatePostQuantizePass(emit_adaptor));

  if (failed(pm.run(module.get()))) {
    const std::string& err = statusHandler.ConsumeStatus().error_message();
    error_reporter->Report("Failed to quantize: %s", err.c_str());
    return kTfLiteError;
  }

  // Export the results to the builder
  std::string result;
  if (tflite::MlirToFlatBufferTranslateFunction(
          module.get(), &result, /*emit_builtin_tflite_ops=*/true,
          /*emit_select_tf_ops=*/true, /*emit_custom_ops=*/true)) {
    error_reporter->Report("Failed to export MLIR to flatbuffer.");
    return kTfLiteError;
  }
  builder->PushFlatBuffer(reinterpret_cast<const uint8_t*>(result.data()),
                          result.size());

  return kTfLiteOk;
}

}  // namespace lite
}  // namespace mlir
