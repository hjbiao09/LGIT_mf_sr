from Vimeo_dataloader import Vimeo
from train_step_ver import YnetTrianer
from train_epoch_ver import YnetTrianer_epoch
from model.MyModel import Mymodel


scale=2
#data_load
Vimeo_train = Vimeo(images_dir="H:/sr_data/vimeo_septuplet/new_sequences", subset="train", scale=scale)
Vimeo_valid = Vimeo(images_dir="H:/sr_data/vimeo_septuplet/new_sequences", subset="test", scale=scale)

train_ds = Vimeo_train.dataset(batch_size=4, random_transform=True)
valid_ds = Vimeo_valid.dataset(batch_size=1, random_transform=False)

#model 및 optim
trainer = YnetTrianer(model=Mymodel(scale=scale), checkpoint_dir=f'./ckpt/Unet_scale_%d'%scale)

#train
trainer.train(train_ds,
              valid_ds.take(1000), #앞 100개만 valid
              steps=500000,
              evaluate_every=100)

# 마지막 최종 psnr 측정
psnr = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr.numpy():.4f}')




