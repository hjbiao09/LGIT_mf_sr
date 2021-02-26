import tensorflow as tf
import cv2
import os

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def evaluate(model, dataset):
    psnr_values = []
    for images, name in dataset:
        lr, hr = images
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)

def save_image(model, dataset, epoch):
    if not os.path.exists("./valid"):
        os.makedir("./valid")
    for images, name in dataset:
        lr, sr = images
        sr_batch = resolve(model, lr)
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        #tensor to numpy and save
        array = tf.keras.preprocessing.image.img_to_array(sr_batch[0])
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./valid/%s"%str(name.numpy()[0])[2:-1], rgb)

        if epoch == 0:
            lr_numpy = lr[0].numpy()
            lr_numpy = cv2.cvtColor(lr_numpy, cv2.COLOR_BGR2RGB)
            sr_numpy = sr[0].numpy()
            sr_numpy = cv2.cvtColor(sr_numpy, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./valid/%s" % str(name.numpy()[0])[2:-5] + "_blur.png", lr_numpy)
            cv2.imwrite("./valid/%s" % str(name.numpy()[0])[2:-5] + "_sharp.png", sr_numpy)


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)