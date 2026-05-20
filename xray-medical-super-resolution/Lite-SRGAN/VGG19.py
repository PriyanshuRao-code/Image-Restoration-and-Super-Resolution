import tensorflow.compat.v2 as tf

from tensorflow.keras import backend
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import utils as keras_utils
data_utils = keras_utils
layer_utils = keras_utils

from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.initializers import GlorotUniform 

WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5"
)
WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

def VGG19(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
):

    if weights not in {"imagenet", None}:
        if not tf.io.gfile.exists(weights):
            raise ValueError(
                "The `weights` argument should be either "
                "`None`, `imagenet`, or path to weights file. "
                f"Received: weights={weights}"
            )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights="imagenet"` with include_top=True, classes must be 1000.'
        )

    if input_shape is None:
        input_shape = (224, 224, 3)
    else:
        if len(input_shape) != 3 or input_shape[-1] != 3:
            raise ValueError(f"Expected input_shape with 3 channels, got {input_shape}")

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor if backend.is_keras_tensor(input_tensor) else Input(tensor=input_tensor, shape=input_shape)

    x = Conv2D(64, (3, 3), padding="same", name="block1_conv1_before_activation")(img_input)
    x = Activation("relu", name="block1_conv1")(x)

    x = Conv2D(64, (3, 3), padding="same", name="block1_conv2_before_activation")(x)
    x = Activation("relu", name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    x = Conv2D(128, (3, 3), padding="same", name="block2_conv1_before_activation")(x)
    x = Activation("relu", name="block2_conv1")(x)

    x = Conv2D(128, (3, 3), padding="same", name="block2_conv2_before_activation")(x)
    x = Activation("relu", name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    x = Conv2D(256, (3, 3), padding="same", name="block3_conv1_before_activation")(x)
    x = Activation("relu", name="block3_conv1")(x)

    x = Conv2D(256, (3, 3), padding="same", name="block3_conv2_before_activation")(x)
    x = Activation("relu", name="block3_conv2")(x)

    x = Conv2D(256, (3, 3), padding="same", name="block3_conv3_before_activation")(x)
    x = Activation("relu", name="block3_conv3")(x)

    x = Conv2D(256, (3, 3), padding="same", name="block3_conv4_before_activation")(x)
    x = Activation("relu", name="block3_conv4")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    x = Conv2D(512, (3, 3), padding="same", name="block4_conv1_before_activation")(x)
    x = Activation("relu", name="block4_conv1")(x)

    x = Conv2D(512, (3, 3), padding="same", name="block4_conv2_before_activation")(x)
    x = Activation("relu", name="block4_conv2")(x)

    x = Conv2D(512, (3, 3), padding="same", name="block4_conv3_before_activation")(x)
    x = Activation("relu", name="block4_conv3")(x)

    x = Conv2D(512, (3, 3), padding="same", name="block4_conv4_before_activation")(x)
    x = Activation("relu", name="block4_conv4")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    x = Conv2D(512, (3, 3), padding="same", name="block5_conv1_before_activation")(x)
    x = Activation("relu", name="block5_conv1")(x)

    x = Conv2D(512, (3, 3), padding="same", name="block5_conv2_before_activation")(x)
    x = Activation("relu", name="block5_conv2")(x)

    x = Conv2D(512, (3, 3), padding="same", name="block5_conv3_before_activation")(x)
    x = Activation("relu", name="block5_conv3")(x)

    x = Conv2D(512, (3, 3), padding="same", name="block5_conv4_before_activation")(x)
    x = Activation("relu", name="block5_conv4")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    model = Model(img_input, x, name="vgg19")

    initializer = GlorotUniform()
    xavier_applied = False
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            w, b = layer.get_weights()
            w = initializer(shape=w.shape)
            b = tf.zeros_like(b) 
            layer.set_weights([w, b])
            xavier_applied=True
    if xavier_applied:
        print(" Xavier initialization applied for VGG19.")
    return model


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="caffe")