import os
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE

class Vimeo:
    def __init__(self, images_dir="H:/sr_data/vimeo_septuplet/new_sequences", subset='train', mode="step", scale=2, frame_num=7):
        self.subset = subset
        self.images_dir = os.path.join(images_dir, self.subset)
        self.mode = mode
        self.sacle=scale
        self.frame_num=frame_num
        self.length = int(self.frame_num//2)

        f = open("./data_list/%s_list.txt" % self.subset)
        self.list = f.readlines()

    def __len__(self):
        return len(self.list)

    def dataset(self, batch_size=16, repeat_count=None, random_transform=False):
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
    def _multi_images_dataset(image_files, frame_num=7):

        ds_list = []
        for i in range(frame_num):
            ds = tf.data.Dataset.from_tensor_slices(image_files[i])
            # tf.data.Dataset.from_tensor_slices 함수에 return 받길 원하는 데이터를 튜플 (data, label) 형태로 넣어서 사용할 수 있습니다.
            ds = ds.map(tf.io.read_file)  # ? 의미 list인데 굳이 더 ? #python open 함수와 비슷
            ds = ds.map(lambda x: tf.image.decode_png(x, channels=3),
                        num_parallel_calls=AUTOTUNE)  # 파일 read후 png로 decode -> 이미지 tensor
            ds_list.append(ds)
        #ds_all = ds_list[0]
        #for i in range(frame_num-1):
        #    ds_all = tf.data.Dataset.zip((ds_all, ds_list[i+1]))
        ds_all = tf.data.Dataset.zip((ds_list[0], ds_list[1], ds_list[2], ds_list[3],
                                     ds_list[4], ds_list[5], ds_list[6]))

        return ds_all

    @staticmethod
    def _images_files_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        # ds = ds.map(tf.io.read_file)
        return ds

    def name_dataset(self):
        ds = self._images_files_dataset(self._image_files())
        return ds

    def lr_dataset(self):
        ds = self._multi_images_dataset(self._lr_image_files(), self.frame_num)
        return ds

    def hr_dataset(self):
        ds = self._images_dataset(self._hr_image_files())
        return ds

    def _lr_image_files(self):
        if self.sacle == 2:
            images_dir = os.path.join(self.images_dir, "lr", "2x")
        if self.sacle == 4:
            images_dir = os.path.join(self.images_dir, "lr", "4x")
        # files_list = os.listdir(images_dir)
        lr_data_list = []
        for file_id in range(len(self.list)):
            file_dir = os.path.join(images_dir, self.list[file_id][:-1])
            lr_list = []

            for i in range(-(self.length), self.length+1):
                lr_list.append(os.path.join(file_dir, "im%s.png" % str(4 + i)))
            lr_data_list.append((lr_list))
        return (lr_data_list)

    def _hr_image_files(self):
        images_dir = os.path.join(self.images_dir,"hr")
        files_list = os.listdir(images_dir)
        return [os.path.join(images_dir, files_list[image_id], "im4.png") for image_id in range(len(self.list))]

    def _image_files(self):
        images_dir = os.path.join(self.images_dir,"lr",)
        files_list = os.listdir(images_dir)
        return [files_list[image_id] for image_id in range(len(files_list))]


    @staticmethod # 모름
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')

# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------


def random_crop(lr_imgs, hr_img, hr_crop_size=256, scale=2, frame_num=7):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_imgs[0])[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale
    lr_imgs_cropped = []
    for i in range(frame_num):
        lr_imgs_cropped.append(lr_imgs[i][lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size])
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_imgs_cropped, hr_img_cropped

def random_flip(lr_imgs, hr_img, frame_num=7):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_imgs, hr_img),
                   lambda: (tf.image.flip_left_right(blur_img),
                            tf.image.flip_left_right(sharp_img)))

def random_rotate(blur_img, sharp_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(blur_img, rn), tf.image.rot90(sharp_img, rn)





