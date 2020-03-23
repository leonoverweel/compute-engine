import numpy as np
import larq_compute_engine.mlir.optimize.calibrator as calibrator
from tensorflow.lite.python import lite_constants as constants


tflite_model = open(
    "/home/tom/Plumerai/compute-engine/experiments/quantization/model_lce.tflite", "rb"
).read()


def representative_dataset_gen():
    for _ in range(2):
        yield [np.zeros((1, 224, 224, 3), dtype=np.float32)]


calibrated_model = calibrator.calibrate(tflite_model, representative_dataset_gen,)
print("Saving calibrated model!")
open(
    "/home/tom/Plumerai/compute-engine/experiments/quantization/model_lce_calibrated.tflite",
    "wb",
).write(calibrated_model)

converted_model = calibrator.calibrate_and_quantize(
    tflite_model, representative_dataset_gen, constants.FLOAT, constants.FLOAT, True,
)

print("Saving quantized model!")
open(
    "/home/tom/Plumerai/compute-engine/experiments/quantization/model_lce_quantized.tflite",
    "wb",
).write(converted_model)
