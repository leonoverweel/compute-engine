import numpy as np
from larq_compute_engine.mlir.quantization.calibration_wrapper import Calibrator


def calibrate_and_quantize(model, dataset_gen, input_type, output_type):
    calibrator = Calibrator(model)
    calibrator.Prepare()
    # Run the images through the model
    for calibration_sample in dataset_gen():
        calibrator.FeedTensor(calibration_sample)
    return calibrator.QuantizeModel(
        np.dtype(input_type.as_numpy_dtype()).num,
        np.dtype(output_type.as_numpy_dtype()).num,
    )
