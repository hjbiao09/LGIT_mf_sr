import os
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE

class GoPro:
    def __init__(self, images_dir="C:/Users/yue95/Desktop/general_deblur/deblur_data/gopro_reset", subset='train', mode="step", scale=2):
        self.subset = subset
        self.images_dir = os.path.join(images_dir, self.subset)
        self.mode = mode
        self.scale = scale

        f = open("./data_list/%s_list.txt" % self.subset)
        self.list = f.readlines()

    def __len__(self):
        return len(self.list)


    def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        if random_transform:
            # ds = ds.map(lambda blur, sharp: random_crop(blur, sharp), num_parallel_calls=AUTOTUNE) #num_parallel_calls? #람다 필요?
            ds = ds.map(random_crop, num_parallel_calls=AUTOTUNE) #이것도 됨 위와 차이 없음.
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)

        # ds = tf.data.Dataset.zip((ds, self.name_dataset())) # 파일 이름 추가
        ds = ds.batch(batch_size)
        if self.mode == "step":
            ds = ds.repeat(repeat_count) #if repeat_count= None or -1 repeated indefinitely
            # step 으로 사용시 repeat(None)
        ds = ds.prefetch(buffer_size=AUTOTUNE) #미리 가져오기
        return ds

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        #tf.data.Dataset.from_tensor_slices 함수에 return 받길 원하는 데이터를 튜플 (data, label) 형태로 넣어서 사용할 수 있습니다.
        ds = ds.map(tf.io.read_file) #? 의미 list인데 굳이 더 ? #python open 함수와 비슷
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE) #파일 read후 png로 decode -> 이미지 tensor
        return ds

    @staticmethod
    def _images_files_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        # ds = ds.map(tf.io.read_file)
        return ds

    def name_dataset(self):
        ds = self._images_files_dataset(self._image_files())
        return ds

    def lr_dataset(self):
        ds = self._images_dataset(self._lr_image_files())
        return ds

    def hr_dataset(self):
        ds = self._images_dataset(self._hr_image_files())
        return ds

    def _lr_image_files(self,num=4,):
        images_dir = os.path.join(self.images_dir, "lr/%sx" % self.scale)
        files_list = os.listdir(images_dir)
        return [os.path.join(images_dir, files_list[image_id], "im%d.png" % num) for image_id in range(len(self.list))]

    def _hr_image_files(self):
        images_dir = os.path.join(self.images_dir, "hr")
        files_list = os.listdir(images_dir)
        return [os.path.join(images_dir, files_list[image_id], "im4.png") for image_id in range(len(self.list))]

    def _image_files(self):
        images_dir = os.path.join(self.images_dir, "hr")
        files_list = os.listdir(images_dir)
        return [files_list[image_id] +"_im4.png" for image_id in range(len(self.list))]

    @staticmethod # 모름
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')

# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------


def random_crop(blur_img, sharp_img, crop_size=256):

    blur_img_shape = tf.shape(blur_img)[:2]
    w = tf.random.uniform(shape=(), maxval=blur_img_shape[1] - crop_size + 1, dtype=tf.int32)
    h = tf.random.uniform(shape=(), maxval=blur_img_shape[0] - crop_size + 1, dtype=tf.int32)

    blur_img_cropped = blur_img[h:h + crop_size, w:w + crop_size]
    sharp_img_cropped = sharp_img[h:h + crop_size, w:w + crop_size]

    return blur_img_cropped, sharp_img_cropped

def random_flip(blur_img, sharp_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5, lambda: (blur_img, sharp_img),
                   lambda: (tf.image.flip_left_right(blur_img),
                            tf.image.flip_left_right(sharp_img)))

def random_rotate(blur_img, sharp_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(blur_img, rn), tf.image.rot90(sharp_img, rn)





