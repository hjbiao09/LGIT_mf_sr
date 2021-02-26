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
from utils import *

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
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        progbar = tf.keras.utils.Progbar(len(train_dataset))
        epoch_start = ckpt.epoch.numpy()

        for epoch_step in range(total_epoch):
            epoch_idx = epoch_start + epoch_step
            self.now = time.perf_counter()
            for i, data in enumerate(train_dataset):
                progbar.update(i + 1)
                images, name = data
                blur, sharp = images

                loss = self.train_step(blur, sharp)  # train 및 loss backward
                loss_mean(loss)
            loss_value = loss_mean.result()
            loss_mean.reset_states()
            duration = time.perf_counter() - self.now
            self.save_image(valid_dataset, epoch_idx)
            psnr_value = self.evaluate(valid_dataset)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=epoch_idx)
                tf.summary.scalar('PSNR', psnr_value, step=epoch_idx)
            current_lr = self.checkpoint.optimizer._decayed_lr(tf.float32).numpy()
            print(
                    f'EPOCH: {epoch_idx}/{total_epoch} loss = {loss_value.numpy(): 3f}, LR = {current_lr: .8f} PSNR = {psnr_value.numpy():.4f} ({duration: .2f})s')

            # ckpt.epoch = epoch_idx
            ckpt.epoch.assign_add(1)
            ckpt.psnr = psnr_value
            ckpt_mgr.save()

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

        # Process the gradients, for example cap them, etc.
        # capped_grads = [MyCapper(g) for g in grads]
        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables) # 그라디언트 생성
        # Ask the optimizer to apply the processed gradients.
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables)) #그라디언트 backward #왜 zip인지는 의문
        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def save_image(self, dataset, epoch):
        return save_image(self.checkpoint.model, dataset, epoch)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint: #세이브 파일 주소
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint) #세이브 파일 로드
            print(f'Model restored from checkpoint at epoch {self.checkpoint.epoch.numpy()}.')

class YnetTrianer_epoch(Trainer):
    def __init__(self, model, checkpoint_dir, lr=PiecewiseConstantDecay(boundaries=[200000,4000000], values=[1e-4, 5e-5, 2.5e-5])): #lr_sch #epoch 일때 lr어떻게 설명해야하나?
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=lr, checkpoint_dir=checkpoint_dir)
