from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Conv2DTranspose, BatchNormalization
from tensorflow.python.keras.layers import Activation, Concatenate
from tensorflow.python.keras.models import Model
import tensorflow as tf

# #다른 형태
# class MyModel(Model):
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.conv1 = Conv2D(32, 3, activation='relu')
#     self.flatten = Flatten()
#     self.d1 = Dense(128, activation='relu')
#     self.d2 = Dense(10, activation='softmax')
#
#   def call(self, x):
#     x = self.conv1(x)
#     x = self.flatten(x)
#     x = self.d1(x)
#     return self.d2(x)
#
# model = MyModel()

def BasicConv(x_in, filters, kernel_size=3, padding='same', bias=True, stride=(1, 1), norm=False, relu=True, transpose=False):
    if transpose:
        x = Conv2DTranspose(filters, kernel_size, strides=stride, padding=padding, use_bias=bias)(x_in)
    else:
        x = Conv2D(filters, kernel_size, strides=stride, padding=padding, use_bias=bias)(x_in)
    if norm:
        x = BatchNormalization()(x)
    if relu:
        x = Activation('relu')(x)
    return x

def resblock(x_in, filters):
    x = BasicConv(x_in, filters=filters, kernel_size=3, relu=True)
    x = BasicConv(x, filters=filters, kernel_size=3, relu=False)
    x = Add()([x, x_in])
    return x


def EBlock(x_in, filters=64, num_res=8, first=False):
    if first:
        x = BasicConv(x_in, filters=filters)
    else:
        x = BasicConv(x_in, filters=filters, stride=(2, 2))
    for i in range(num_res):
        x = resblock(x, filters=filters)
    return x


def DBlock(x_in, filters=64, num_res=8, last=False):
    x = x_in

    for i in range(num_res):
        x = resblock(x, filters=filters)

    if last:
        x = BasicConv(x, filters=3, relu=False)
    else:
        x = BasicConv(x, kernel_size=4, filters=filters//2, transpose=True, stride=(2, 2))

    return x

def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0

def denormalize_255(x):
    return x * 255.0

def Ynet():
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(x_in)

    x_size = tf.shape(x)
    x_up = tf.image.resize(x, size=[x_size[1]*2, x_size[2]*2])

    res_num_encoder = 6
    base_channels = 32
    x_e1 = EBlock(x_up, filters=base_channels, num_res=res_num_encoder, first=True)
    x_e2 = EBlock(x_e1, filters=2 * base_channels, num_res=res_num_encoder)
    x_e3 = EBlock(x_e2, filters=4 * base_channels, num_res=res_num_encoder)

    res_num_decoder = 6
    x_d3_1 = DBlock(x_e3, filters=4 * base_channels, num_res=res_num_decoder)
    x_c2_1 = BasicConv(Concatenate(axis=3)([x_d3_1, x_e2]), 2 * base_channels, kernel_size=1)
    x_d2_1 = DBlock(x_c2_1, filters=2 * base_channels, num_res=res_num_decoder)
    x_c1_1 = BasicConv(Concatenate(axis=3)([x_d2_1, x_e1]), base_channels, kernel_size=1)
    x_out_1 = DBlock(x_c1_1, filters=base_channels, num_res=res_num_decoder, last=True)

    # x_d3_2 = DBlock(x_e3, filters=4 * base_channels, num_res=res_num_decoder)
    # x_c2_2 = BasicConv(Concatenate(axis=3)([x_d3_2, x_e2]), 2 * base_channels, kernel_size=1)
    # x_d2_2 = DBlock(x_c2_2, filters=2 * base_channels, num_res=res_num_decoder)
    # x_c1_2 = BasicConv(Concatenate(axis=3)([x_d2_2, x_e1]), base_channels, kernel_size=1)
    # x_out_2 = DBlock(x_c1_2, filters=base_channels, num_res=res_num_decoder, last=True)

    deblur = x_up + x_out_1 #+ x_out_2
    deblur = Lambda(denormalize_255)(deblur)
    return Model(x_in, deblur, name="YNet")










