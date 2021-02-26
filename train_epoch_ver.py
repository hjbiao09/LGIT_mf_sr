import time
import tensorflow as tf

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tqdm import tqdm
import cv2
import tensorflow_addons as tfa
import datetime

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

def save_image(model, dataset):
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

        lr_numpy = lr[0].numpy()
        lr_numpy = cv2.cvtColor(lr_numpy, cv2.COLOR_BGR2RGB)
        sr_numpy = sr[0].numpy()
        sr_numpy = cv2.cvtColor(sr_numpy, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./valid/%s" % str(name.numpy()[0])[2:-5]+"_blur.png", lr_numpy)
        cv2.imwrite("./valid/%s" % str(name.numpy()[0])[2:-5]+"_sharp.png", sr_numpy)

def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/ynet'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0),
                                              psnr=tf.Variable(-1.0), #굳이 tf 형태로?
                                              optimizer=Adam(learning_rate),
                                              model=model) #체크포인트 model저장
        self.model_save = tf.train.Checkpoint(model=model)
        #이것을 .save() .restore()
        # checkpoint.save('./save/model.ckpt')
        # 원하는 파라미터 저장 e.g step, psnr
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)
        # 단순 checkpoint로 저장시 나중에 대량의 save파일이 존재 할 수 있음.
        # 그걸 방지하지 위한 checkpointmanager 최후의 max_to_keep 개의 파일만 저장해줌.
        # 사용시 바로 self.checkpoint_manager.save() 혹은 self.checkpoint_manager.save(checkpoint_number=100)

        self.restore() #해당 모델 restore() 아래 함수있음.

    @property #?
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, total_epoch):
        loss_mean = Mean()

        #tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + "/train"

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        # length = len(list(train_dataset))
        progbar = tf.keras.utils.Progbar(len(train_dataset))
        epoch_idx = ckpt.epoch.numpy()

        for epoch_idx in tqdm(range(total_epoch)):
            self.now = time.perf_counter()
            # for images, name in tqdm(train_dataset.take(steps - ckpt.step.numpy())): #step 일때
            for i, data in enumerate(train_dataset):
                progbar.update(i + 1)
                images, name = data
                blur, sharp = images

                loss = self.train_step(blur, sharp)  # train 및 loss backward
                loss_mean(loss)
            loss_value = loss_mean.result()
            loss_mean.reset_states()
            duration = time.perf_counter() - self.now
            self.save_image(valid_dataset)
            psnr_value = self.evaluate(valid_dataset)

            #tensorboard 저장 현재 tensorboard에서 안나옴 확인 필요.
            # valid_log_dir = 'logs/gradient_tape/' + current_time + "/valid"
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            tf.summary.scalar('loss', loss_value, step=epoch_idx)
            tf.summary.scalar('PSNR', psnr_value, step=epoch_idx)

            print(
                    f'{epoch_idx}/{total_epoch}: loss = {loss_value.numpy(): 3f}, PSNR = {psnr_value.numpy():.4f} ({duration: .2f})s')
            ckpt.epoch = epoch_idx
            ckpt.psnr = psnr_value
            ckpt_mgr.save(checkpoint_number=ckpt.step + evaluate_every)

            self.now = time.perf_counter()


    @tf.function
    def train_step(self, blur, sharp): #loss backword
        with tf.GradientTape() as tape:
            blur = tf.cast(blur, tf.float32)
            sharp = tf.cast(sharp, tf.float32)

            deblur = self.checkpoint.model(blur, training=True) #여기서 weight sharing 할때 모델 공유?
            # e.g.
            # y = model(x)
            # x = model(x)
            # loss = x + y
            loss_value = self.loss(sharp, deblur)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables) # 그라디언트 생성
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables)) #그라디언트 적융 #왜 zip인지는 의문
        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def save_image(self, dataset):
        return save_image(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint: #세이브 파일 주소
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint) #세이브 파일 로드
            print(f'Model restored from checkpoint at stop {self.checkpoint.step.numpy()}.')

class YnetTrianer_epoch(Trainer):
    def __init__(self, model, checkpoint_dir, lr=PiecewiseConstantDecay(boundaries=[200000,400000], values=[1e-4, 5e-5, 2.5e-5])): #lr_sch
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=lr, checkpoint_dir=checkpoint_dir)


