import numpy as np
from larq_compute_engine.mlir.optimize.calibration_wrapper import Calibrator


def calibrate(model, dataset_gen):
    calibrator = Calibrator(model)
    calibrator.Prepare()
    # Run the images through the model
    for calibration_sample in dataset_gen():
        calibrator.FeedTensor(calibration_sample)
    return calibrator.GetCalibrated()


def calibrate_and_quantize(
    model, dataset_gen, input_type, output_type, allow_float,
):
    calibrator = Calibrator(model)
    calibrator.Prepare()
    # Run the images through the model
    for calibration_sample in dataset_gen():
        calibrator.FeedTensor(calibration_sample)
    return calibrator.QuantizeModel(
        np.dtype(input_type.as_numpy_dtype()).num,
        np.dtype(output_type.as_numpy_dtype()).num,
        allow_float,
    )
