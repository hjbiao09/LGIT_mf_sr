from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Conv2DTranspose, BatchNormalization
from tensorflow.python.keras.layers import Activation, Concatenate
from tensorflow.python.keras.models import Model


class BasicConv(Model):
    def __init__(self,filters, kernel_size=3, padding='same', bias=True, stride=(1, 1), norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if transpose:
            x = Conv2DTranspose(filters, kernel_size, strides=stride, padding=padding, use_bias=bias)

