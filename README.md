# MEVO-GAN
基于多尺度生成对抗进化网络的水下图像增强

## Dependences
Run  `pip install requirements`

## Training
Run ` python train.py --model EC_gan
` for training the proposed method（Regarding the option settings, you can view the options folder by yourself）.
## Testing
Run `python test.py --model test ` for generating test data.
you can download the trained model from 
## Acknowledgments
Baseline codes from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
