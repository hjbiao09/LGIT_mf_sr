import tensorflow as tf
import cv2
import os

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr1, lr2, lr3, lr4, lr5, lr6, lr7):
    lr1 = tf.cast(lr1, tf.float32)
    lr2 = tf.cast(lr2, tf.float32)
    lr3 = tf.cast(lr3, tf.float32)
    lr4 = tf.cast(lr4, tf.float32)
    lr5 = tf.cast(lr5, tf.float32)
    lr6 = tf.cast(lr6, tf.float32)
    lr7 = tf.cast(lr7, tf.float32)
    sr_batch = model(inputs=[lr1, lr2, lr3, lr4, lr5, lr6, lr7])
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def evaluate(model, dataset):
    psnr_values = []
    for i, images in enumerate(dataset):
        lr1, lr2, lr3, lr4, lr5, lr6, lr7, hr = images
        sr = resolve(model, lr1, lr2, lr3, lr4, lr5, lr6, lr7)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
        print('\r%04d images are evaluated.' % i, end=' ')
    return tf.reduce_mean(psnr_values)

def save_image(model, dataset, step, first_step=100):
    if not os.path.exists("./valid"):
        os.mkdir("./valid")
    for i, images in enumerate(dataset):
        lr1, lr2, lr3, lr4, lr5, lr6, lr7, hr = images
        sr_batch = resolve(model, lr1, lr2, lr3, lr4, lr5, lr6, lr7)
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        #tensor to numpy and save
        array = tf.keras.preprocessing.image.img_to_array(sr_batch[0])
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./valid/%04d_results.png"%i, rgb) # b' ~~~ \ 로 읽음. need to fix.
        print('\r%04d images are restorated.'%i, end=' ')
        if step == first_step:
            lr_numpy = lr4[0].numpy()
            lr_numpy = cv2.cvtColor(lr_numpy, cv2.COLOR_BGR2RGB)
            sr_numpy = hr[0].numpy()
            sr_numpy = cv2.cvtColor(sr_numpy, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./valid/%04d" % i + "_lr.png", lr_numpy)
            cv2.imwrite("./valid/%04d" % i + "_hr.png", sr_numpy)


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)