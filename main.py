from gopro_dataloader import GoPro
from train_step_ver import YnetTrianer
from train_epoch_ver import YnetTrianer_epoch
from model.YNet import Ynet

# #data_load
# GoPro_train = GoPro(images_dir="C:/Users/yue95/Desktop/general_deblur/deblur_data/gopro_reset", subset="train")
# GoPro_valid = GoPro(images_dir="C:/Users/yue95/Desktop/general_deblur/deblur_data/gopro_reset", subset="valid")
#
# train_ds = GoPro_train.dataset(batch_size=8, random_transform=True)
# valid_ds = GoPro_valid.dataset(batch_size=1, random_transform=False)
#
# #model 및 optim
# trainer = YnetTrianer(model=Ynet(), checkpoint_dir=f'./ckpt/ynet_two_decoders_step')
#
# #train
# trainer.train(train_ds,
#               valid_ds.take(100),
#               steps=500000,
#               evaluate_every=1000)

#data_load
GoPro_train = GoPro(images_dir="C:/Users/yue95/Desktop/general_deblur/deblur_data/gopro_reset", subset="train", mode="epoch")
GoPro_valid = GoPro(images_dir="C:/Users/yue95/Desktop/general_deblur/deblur_data/gopro_reset", subset="valid", mode="epoch")

train_ds = GoPro_train.dataset(batch_size=8, random_transform=True)
valid_ds = GoPro_valid.dataset(batch_size=1, random_transform=False)

#model 및 optim
trainer = YnetTrianer_epoch(model=Ynet(), checkpoint_dir=f'./ckpt/ynet_two_decoders_epoch')
#train
trainer.train(train_ds,
              valid_ds.take(100),
              total_epoch=1000)

# trainer.restore()
# 마지막 최종 psnr 측정
psnr = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr.numpy():.4f}')




