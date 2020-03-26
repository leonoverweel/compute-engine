import numpy as np
import larq_compute_engine.mlir.quantization.calibrator as calibrator
from tensorflow.lite.python import lite_constants as constants


tflite_model = open("/tmp/quicknet_v3.tflite", "rb").read()


def representative_dataset_gen():
    for _ in range(2):
        yield [np.zeros((1, 224, 224, 3), dtype=np.float32)]


converted_model = calibrator.calibrate_and_quantize(
    tflite_model, representative_dataset_gen, constants.FLOAT, constants.FLOAT
)

print("Saving quantized model!")
open("/tmp/quicknet_v3_quantized.tflite", "wb",).write(converted_model)
