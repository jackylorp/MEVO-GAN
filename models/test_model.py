#coding=utf-8
from .base_model import BaseModel
from . import networks
import cv2
from util import util
import torch
import numpy as np
from skimage import  exposure
import math
from .cal_ssim import SSIM
import numpy as np
from skimage import color
class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test_txt instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test_txt phase. You can use this flag to add training-specific or test_txt-specific options.

        Returns:
            the modified parser.

        The model can only be used during test_txt time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='unaligned')
        parser.add_argument('--model_suffix', type=str, default='_A', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.count = 0
        self.psnr = 0
        self.Ssim = 0
        self.uiqm = 0
        self.uciqe = 0
        # specify the training losses you want to print out. The training/test_txt scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test_txt scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B','fake']
        # specify the models you want to save to the disk. The training/test_txt scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        # self.real_A = input['A','B'].to(self.device)
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B'].to(self.device)
        # self.image_paths = input['A_paths','B_paths']
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    
    
    def getUCIQE(self,image_tensor):
      

      # 转换为BGR颜色空间
      img_BGR = (image_tensor * 255).astype(np.uint8)

      # 转换为LAB颜色空间
      img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
      img_LAB = np.array(img_LAB, dtype=np.float64)
      # Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576 according to paper.
      coe_Metric = [0.4680, 0.2745, 0.2576]
      
      img_lum = img_LAB[:,:,0]/255.0
      img_a = img_LAB[:,:,1]/255.0
      img_b = img_LAB[:,:,2]/255.0

      # item-1
      chroma = np.sqrt(np.square(img_a)+np.square(img_b))
      sigma_c = np.std(chroma)

      # item-2
      img_lum = img_lum.flatten()
      sorted_index = np.argsort(img_lum)
      top_index = sorted_index[int(len(img_lum)*0.99)]
      bottom_index = sorted_index[int(len(img_lum)*0.01)]
      con_lum = img_lum[top_index] - img_lum[bottom_index]

      # item-3
      chroma = chroma.flatten()
      sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum!=0)
      avg_sat = np.mean(sat)

      uciqe = sigma_c*coe_Metric[0] + con_lum*coe_Metric[1] + avg_sat*coe_Metric[2]
      return uciqe

    def calculate_uciqe(self, image):
          image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
         # 将图像转换为Lab色彩空间
          lab_image = color.rgb2lab(image)

          # UCIQE参数
          c1 = 0.4680
          c2 = 0.2745
          c3 = 0.2576

          # 提取亮度通道
          l_channel = lab_image[:, :, 0]

          # 计算色度
          chroma = np.sqrt(lab_image[:, :, 1] ** 2 + lab_image[:, :, 2] ** 2)

          # 计算UC和SC
          uc = np.mean(chroma)
          sc = np.sqrt(np.mean((chroma - uc) ** 2))

          # 提取亮度通道的前1%像素
          top_percentage = 0.01
          num_pixels = l_channel.size
          num_top_pixels = int(top_percentage * num_pixels)
          sorted_l_channel = np.sort(l_channel.flatten())
          top_l_values = sorted_l_channel[-num_top_pixels:]

          # 计算CONL
          conl = np.mean(top_l_values) - np.mean(l_channel)

          # 计算饱和度
          satur = chroma / (l_channel + 1e-10)  # 避免除以零
          us = np.mean(satur)

          # 计算UCIQE
          uciqe = c1 * sc + c2 * conl + c3 * us
          

          return uciqe

    
    def forward(self):
        """Run forward pass."""
      
        self.fake = self.netG(self.real_A)  # G(real)
        

        self.fake1 = util.tensor2im(self.fake)
  
        self.Uciqe = self.getUCIQE(self.fake1)
        
        # self.ssim_fake = self.calculate_ssim(img,fake)
        # self.psnr_fakeB = self.PSNR(img,fake)
        self.count += 1
    
    
    
        self.uciqe += self.Uciqe
       
        
        print('avg_uciqe=',self.uciqe/self.count)

        # self.fake3 = self.netG(fake1)

        
    def optimize_parameters(self):
        """No optimization for test_txt model."""
        pass
