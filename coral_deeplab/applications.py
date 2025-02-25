# MIT License

# Copyright (c) 2021 xadrianzetx

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from coral_deeplab._blocks import deeplab_aspp_module
from coral_deeplab._blocks import deeplabv3_decoder
from coral_deeplab._blocks import deeplabv3plus_decoder
from coral_deeplab._encoders import mobilenetv2


__all__ = ["CoralDeepLabV3", "CoralDeepLabV3Plus"]


def CoralDeepLabV3(
    input_shape: tuple = (513, 513, 3), alpha: float = 1.0, n_classes: int = 30, **kwargs
) -> tf.keras.Model:
    """DeepLab v3 implementation compilable to coral.ai Edge TPU.

    Implementation follows original paper as close as possible, and
    compiles to TPU up to decoder conv layer providing significant
    speedup over CPU inference time.

    MobileNetV2 is used as encoder, but last 3 blocks had been modified
    to use atrous convolution in order to preserve spatial resolution.

    Arguments
    ---------
    input_shape : tuple, default=(513, 513, 3)
        Input tensor shape.

    alpha : float, default=1.0
        Float between 0. and 1.
        MobileNetV2 depth multiplier.

    n_classes : int, default=30
        Number of segmentation classes.
        By default set to cityscapes dayaset
        number of class labels.

    Returns
    -------
    model : tf.keras.Model
        DeepLabV3 keras model instance.

    References
    ----------
    - [1] https://arxiv.org/pdf/1706.05587.pdf
    - [2] https://coral.ai/products/

    Notes
    -----
    There is no last activation layer. Model outputs logits.
    Last layer in the decoder (bilinear upsampling) has been
    removed for performance reasons, making this model OS16.

    Examples
    --------
    >>> import coral_deeplab as cdl
    >>> model = cdl.applications.CoralDeepLabV3()
    >>> print(model.name)
    'CoralDeepLabV3'
    """

    if np.argmin(input_shape) == 0:
        # assuming channels always
        # gonna be smallest number
        raise ValueError("Channels-first not supported.")

    if input_shape[0] != input_shape[1]:
        raise ValueError("Non square inputs not supported.")

    inputs = Input(shape=input_shape)
    aspp_in = mobilenetv2(inputs, alpha)
    aspp_out = deeplab_aspp_module(aspp_in)
    outputs = deeplabv3_decoder(aspp_out, n_classes)
    name = "CoralDeeplabV3"
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model


def CoralDeepLabV3Plus(
    input_shape: tuple = (513, 513, 3), alpha: float = 1.0, n_classes: int = 30, **kwargs
) -> tf.keras.Model:
    """DeepLabV3 Plus implementation compilable to coral.ai Edge TPU.

    Implementation follows original paper as close as possible, and
    compiles to TPU up to decoder conv layer providing significant
    speedup over CPU inference time.

    MobileNetV2 is used as encoder, but last 3 blocks had been modified
    to use atrous convolution in order to preserve spatial resolution.

    Arguments
    ---------
    input_shape : tuple, default=(513, 513, 3)
        Input tensor shape.

    alpha : float, default=1.0
        Float between 0. and 1.
        MobileNetV2 depth multiplier.

    n_classes : int, default=30
        Number of segmentation classes.
        By default set to cityscapes dayaset
        number of class labels.

    Returns
    -------
    model : tf.keras.Model
        DeepLabV3Plus keras model instance.

    References
    ----------
    - [1] https://arxiv.org/pdf/1802.02611.pdf
    - [2] https://coral.ai/products/

    Notes
    -----
    There is no last activation layer. Model outputs logits.
    Last layer in the decoder (bilinear upsampling) has been
    removed for performance reasons, but one in decoder is still
    present making this model OS4 (output size is roughly 4x smaller
    than input size).

    Examples
    --------
    >>> import coral_deeplab as cdl
    >>> model = cdl.applications.CoralDeepLabV3Plus()
    >>> print(model.name)
    'CoralDeepLabV3Plus'
    """

    encoder = CoralDeepLabV3(input_shape, alpha)
    encoder_last = encoder.get_layer("concat_projection/relu")
    encoder_skip = encoder.get_layer("expanded_conv_3/expand/relu")
    outputs = deeplabv3plus_decoder(encoder_last.output, encoder_skip.output, n_classes)
    name = "CoralDeeplabV3Plus"
    model = tf.keras.Model(inputs=encoder.inputs, outputs=outputs, name=name)

    return model
