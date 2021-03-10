from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Concatenate
from tensorflow.python.keras.models import Model

from model.common import normalize, denormalize, pixel_shuffle


def edsr(scale=4, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x1 = Input(shape=(None, None, 3))
    x2 = Input(shape=(None, None, 3))
    x3 = Input(shape=(None, None, 3))
    x4 = Input(shape=(None, None, 3))
    x5 = Input(shape=(None, None, 3))
    x6 = Input(shape=(None, None, 3))
    x7 = Input(shape=(None, None, 3))
    lr1 = Lambda(normalize)(x1)
    lr2 = Lambda(normalize)(x2)
    lr3 = Lambda(normalize)(x3)
    lr4 = Lambda(normalize)(x4)
    lr5 = Lambda(normalize)(x5)
    lr6 = Lambda(normalize)(x6)
    lr7 = Lambda(normalize)(x7)
    x = Concatenate(axis=3)([lr1, lr2, lr3, lr4, lr5, lr6, lr7])

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(inputs=[x1, x2, x3, x4, x5, x6, x7], outputs=x, name="edsr")


def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x