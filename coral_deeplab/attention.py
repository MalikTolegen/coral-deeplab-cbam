import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.shared_dense_1 = layers.Dense(channel // self.reduction_ratio, activation='relu')
        self.shared_dense_2 = layers.Dense(channel)

    def call(self, inputs):
        avg_out = self.shared_dense_2(self.shared_dense_1(self.avg_pool(inputs)))
        max_out = self.shared_dense_2(self.shared_dense_1(self.max_pool(inputs)))
        scale = tf.nn.sigmoid(avg_out + max_out)
        scale = tf.reshape(scale, [-1, 1, 1, scale.shape[-1]])
        return inputs * scale

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        scale = self.conv(concat)
        return inputs * scale

class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x
