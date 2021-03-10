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
from utils_iter import *

def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/unet'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
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
        self.checkpoint_dir=checkpoint_dir
        # 단순 checkpoint로 저장시 나중에 대량의 save파일이 존재 할 수 있음.
        # 그걸 방지하지 위한 checkpointmanager 최후의 max_to_keep 개의 파일만 저장해줌.
        # 사용시 바로 self.checkpoint_manager.save() 혹은 self.checkpoint_manager.save(checkpoint_number=100)

        self.restore() #해당 모델 restore() 아래 함수있음.

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        train_log_dir = 'logs/gradient_tape/%s/train'%self.checkpoint_dir.split("/")[-1]
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        progbar = tf.keras.utils.Progbar((steps - ckpt.step.numpy()))
        self.now = time.perf_counter()
        # for images, name in tqdm(train_dataset.take(steps - ckpt.step.numpy())): #둘다 크게 상관 없음
        for i, data in enumerate(train_dataset.take(steps - ckpt.step.numpy())):
            progbar.update(i + 1)
            images = data
            lr1, lr2, lr3, lr4, lr5, lr6, lr7, hr = images
            # lr4, hr = images
            ckpt.step.assign_add(1) # step += step
            step = ckpt.step.numpy()
            loss = self.train_step(lr1, lr2, lr3, lr4, lr5, lr6, lr7, hr)  # train 및 loss backward
            loss_mean(loss)
            if ckpt.step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()
                duration = time.perf_counter() - self.now
                self.save_image(valid_dataset, ckpt.step.numpy(), evaluate_every, folder_name=self.checkpoint_dir.split("/")[-1])
                psnr_value = self.evaluate(valid_dataset)

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_value, step=ckpt.step.numpy())
                    tf.summary.scalar('PSNR', psnr_value, step=ckpt.step.numpy())
                current_lr = self.checkpoint.optimizer._decayed_lr(tf.float32).numpy()
                print(
                    f'EPOCH: {step}/{steps} loss = {loss_value.numpy(): 3f}, LR = {current_lr: .8f} PSNR = {psnr_value.numpy():.7f} ({duration:.2f})s')
                ckpt.psnr = psnr_value
                ckpt_mgr.save(checkpoint_number=ckpt.step.numpy())

                self.now = time.perf_counter()

    #텐서플로 2에서는 즉시 실행(eager execution)이 기본적으로 활성화되어 있습니다. 직관적이고 유연한 사용자 인터페이스를 제공하지만 성능과 배포에 비용이 더 듭니다(하나의 연산을 실행할 때는 훨씬 간단하고 빠릅니다).
    #성능을 높이고 이식성이 좋은 모델을 만들려면 tf.function을 사용해 그래프로 변환하세요.
    @tf.function
    def train_step(self, lr1, lr2, lr3, lr4, lr5, lr6, lr7, hr): #loss backword
        with tf.GradientTape() as tape:
            lr1 = tf.cast(lr1, tf.float32)
            lr2 = tf.cast(lr2, tf.float32)
            lr3 = tf.cast(lr3, tf.float32)
            lr4 = tf.cast(lr4, tf.float32)
            lr5 = tf.cast(lr5, tf.float32)
            lr6 = tf.cast(lr6, tf.float32)
            lr7 = tf.cast(lr7, tf.float32)
            hr = tf.cast(hr, tf.float32)

            result = self.checkpoint.model(inputs=[lr1, lr2, lr3, lr4, lr5, lr6, lr7], training=True) #여기서 weight sharing 할때 모델 공유?
            # e.g.
            # y = model(x)
            # x = model(x)
            # loss = x + y

            loss_value = self.loss(result, hr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables) # 그라디언트 생성
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables)) #그라디언트 적융 #왜 zip인지는 의문
        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def save_image(self, dataset,step, first_step, folder_name):
        return save_image(self.checkpoint.model, dataset, step, first_step=first_step, folder_name=folder_name)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint: #세이브 파일 주소
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint) #세이브 파일 로드
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

class YnetTrianer(Trainer):
    def __init__(self, model, checkpoint_dir, lr=PiecewiseConstantDecay(boundaries=[100000,300000], values=[1e-4, 5e-5, 2.5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=lr, checkpoint_dir=checkpoint_dir)


