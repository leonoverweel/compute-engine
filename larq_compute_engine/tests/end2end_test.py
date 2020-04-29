import sys
import pytest
import larq as lq
import larq_zoo as lqz
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# import tensorflow_datasets as tfds
import numpy as np

from larq_compute_engine.mlir.python.converter import convert_keras_model

# from larq_compute_engine.tests._end2end_verify import run_model


def toy_model(**kwargs):
    def block(padding, pad_values, activation):
        def dummy(x):
            shortcut = x
            x = lq.layers.QuantConv2D(
                filters=32,
                kernel_size=3,
                padding=padding,
                pad_values=pad_values,
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                use_bias=False,
                activation=activation,
            )(x)
            x = tf.keras.layers.BatchNormalization(
                gamma_initializer=tf.keras.initializers.RandomNormal(1.0),
                beta_initializer="uniform",
            )(x)
            return tf.keras.layers.add([x, shortcut])

        return dummy

    img_input = tf.keras.layers.Input(shape=(224, 224, 3))
    out = img_input
    out = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(out)

    # Test zero-padding
    out = block("same", 0.0, "relu")(out)
    # Test one-padding
    out = block("same", 1.0, "relu")(out)
    # Test no activation function
    out = block("same", 1.0, None)(out)

    out = tf.keras.layers.GlobalAvgPool2D()(out)
    return tf.keras.Model(inputs=img_input, outputs=out)


def quant_toy(**kwargs):
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = np.expand_dims(train_images / 255.0, -1)
    test_images = np.expand_dims(test_images / 255.0, -1)

    # Define the model architecture.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=12, kernel_size=(3, 3), activation=tf.nn.relu
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )

    # Train the digit classification model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        train_images, train_labels, epochs=1, validation_split=0.1,
    )

    quantize_model = tfmot.quantization.keras.quantize_model

    # q_aware stands for for quantization aware.
    q_aware_model = quantize_model(model)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    train_images_subset = train_images[0:1000]  # out of 60000
    train_labels_subset = train_labels[0:1000]

    q_aware_model.fit(
        train_images_subset,
        train_labels_subset,
        batch_size=500,
        epochs=1,
        validation_split=0.1,
    )

    return q_aware_model


def quant_fcnn(**kwargs):
    fcnn = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28)),
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )
    return tfmot.quantization.keras.quantize_model(fcnn)


def quant(x):
    return tf.quantization.fake_quant_with_min_max_vars(x, -3.0, 3.0)


def baseline_model(**kwargs):
    img = tf.keras.layers.Input(shape=(32, 32, 3))
    x = img
    x = lq.layers.QuantConv2D(
        12, 3, input_quantizer=quant, kernel_quantizer=quant, activation="relu"
    )(x)
    x = lq.layers.QuantConv2D(
        12, 3, input_quantizer=quant, kernel_quantizer=quant, activation="relu"
    )(x)
    x = quant(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = lq.layers.QuantDense(
        10, input_quantizer=quant, kernel_quantizer=quant, activation=quant
    )(x)
    x = tf.keras.layers.Activation("softmax")(x)
    return tf.keras.Model(img, x)


def binary_model(**kwargs):
    img = tf.keras.layers.Input(shape=(32, 32, 3))
    x = img
    x = quant(x)
    x = lq.layers.QuantConv2D(
        12, 3, input_quantizer="ste_sign", kernel_quantizer="ste_sign", activation=quant
    )(x)
    x = lq.layers.QuantConv2D(
        12, 3, input_quantizer="ste_sign", kernel_quantizer="ste_sign", activation=quant
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = lq.layers.QuantDense(
        10, input_quantizer=quant, kernel_quantizer=quant, activation=quant
    )(x)
    x = tf.keras.layers.Activation("softmax")(x)
    return tf.keras.Model(img, x)


def feathernet(**kwargs):
    with tfmot.quantization.keras.quantize_scope():
        return tf.keras.models.load_model("/tmp/feathernet_quantized.h5")


def preprocess(data):
    return lqz.preprocess_input(data["image"])


@pytest.mark.parametrize(
    "model_cls", [binary_model],
)
def test_simple_model(model_cls):
    model = model_cls(weights=None)
    model_lce = convert_keras_model(model)

    with open("/tmp/testmodel.tflite", "wb") as f:
        f.write(model_lce)

    # # Test on the flowers dataset
    # dataset = (
    #     tfds.load("oxford_flowers102", split="validation")
    #     .map(preprocess)
    #     .shuffle(256)
    #     .batch(10)
    #     .take(1)
    # )
    # inputs = next(tfds.as_numpy(dataset))
    #
    # outputs = model(inputs).numpy()
    # for input, output in zip(inputs, outputs):
    #     for actual_output in run_model(model_lce, list(input.flatten())):
    #         np.testing.assert_allclose(actual_output, output, rtol=0.001, atol=0.25)
    #
    # # Test on some random inputs
    # input_shape = (10, *model.input.shape[1:])
    # inputs = np.random.uniform(-1, 1, size=input_shape).astype(np.float32)
    # outputs = model(inputs).numpy()
    # for input, output in zip(inputs, outputs):
    #     for actual_output in run_model(model_lce, list(input.flatten())):
    #         np.testing.assert_allclose(actual_output, output, rtol=0.001, atol=0.25)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
