import itertools
import math
from tkinter import *

import cv2
import numpy as np
import torch

from util import util
from util.image_pool import ImagePool

from . import networks
from .base_model import BaseModel
from .cal_ssim import SSIM
import copy 
import torch.nn as nn
import pdb
import torchvision.models as models
    
class ECGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=12.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.6, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            '''vanilla'''
            parser.add_argument('--g_loss_mode', nargs='*', default=['wgan','lsgan','rsgan'], help='lsgan | nsgan | vanilla | wgan | hinge | rsgan')
            parser.add_argument('--d_loss_mode', type=str, default='lsgan', help='lsgan | nsgan | vanilla | wgan | hinge | rsgan') 
            parser.add_argument('--which_D', type=str, default='S', help='Standard(S) | Relativistic_average (Ra)') 
            parser.add_argument('--lambda_f', type=float, default=0.1, help='the hyperparameter that balance Fq and Fd')
            parser.add_argument('--candi_num', type=int, default=3, help='# of survived candidatures in each evolutinary iteration.')
            parser.add_argument('--eval_size', type=int, default=64, help='batch size during each evaluation.')
        return parser
    
    def __init__(self, opt):
        
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','vggA','vggB']#'perceptual_loss_A'
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        self.Calculate_n = 0
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'G_A', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'G_B', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # self.netG_C = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'G_B', opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_C = networks.define_G(opt.input_nc, opt.output_nc, opt.nuf, opt.Unet, opt.norm,not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                             opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                             opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
           
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define G mutations 
            self.G_mutations = []
            for g_loss in opt.g_loss_mode: 
                self.G_mutations.append(networks.GANLoss(g_loss, 'G', opt.which_D).to(self.device))
                #print("!!!!",g_loss)
                #self.G_mutations.append(networks.GANLoss(g_loss, 'G', 'S').to(self.device))
            # define loss functions
            self.criterionGANnoevo = networks.GANLoss_NOEVO(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionGAN = networks.GANLoss(g_loss, 'G', opt.which_D).to(self.device)  # define GAN loss.#默认会用最后一个损失
            ###要改
            self.criterionVGGA = networks.VGGLoss(self.device)
            self.criterionVGGB = networks.VGGLoss(self.device)
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionD = networks.GANLoss('lsgan', 'D',opt.which_D).to(self.device)#这里D_A D_B 都一样
            self.mseloss = torch.nn.MSELoss()
            self.criterionIdt = torch.nn.L1Loss()
            # self.criterionPerceptual = PerceptualLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),lr=opt.lr,betas=(opt.beta1,0.999)) 
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=0.00001, betas=(opt.beta1, 0.999))#这里抄的有问题
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),lr=opt.lr,betas=(opt.beta1,0.999))
        # Evolutinoary candidatures setting (init) 
        self.G_candis = [[],[]] 
        self.optG_candis = [] 
        for i in range(opt.candi_num): 
            self.G_candis[0].append(copy.deepcopy(self.netG_A.state_dict()))
            self.G_candis[1].append(copy.deepcopy(self.netG_B.state_dict()))
            self.optG_candis.append(copy.deepcopy(self.optimizer_G.state_dict()))
        #self.N =int(np.trunc(np.sqrt(min(opt.batch_size, 64))))
        #N means the number of images in a row or column
       
    
        #self.eval_size = max(math.ceil((opt.batch_size * opt.D_iters) / opt.candi_num), opt.eval_size)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_A = self.real_A.transforms.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.65, hue=0.3)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    ## PSNR 评价指标，使用的是均方误差
    def PSNR(self,img1, img2):
        b, _, _, _ = img1.shape
        self.img1 = np.clip(img1, 0, 255)
        self.img2 = np.clip(img2, 0, 255)
        self.mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
        if self.mse == 0:
            return 100
        self.PIXEL_MAX = 1
        return 20 * math.log10(self.PIXEL_MAX / math.sqrt(self.mse))
    
 




   
   
   





    def forwardnoevo(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""


        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        
        # self.fake_B=self.netG_A(self.fake_B1)
          # G_B(G_A(A))

        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        self.rec_A = self.netG_B(self.fake_B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

        self.ssim=SSIM().cuda()
        output = self.fake_B.data.cpu().clamp(-1, 1)
        output = output 
        gt = self.real_B.data.cpu() 
        self.ssim_fakeB = self.ssim(output, gt).cpu().squeeze().detach().numpy()
        self.psnr_fakeB = self.PSNR(output.data.cpu().numpy() * 255, gt.data.cpu().numpy() * 255)
       
       
        print('ssim_noise=',self.ssim_fakeB)
        print('psnr_fakeB= ', self.psnr_fakeB)
       
    
    def forward(self, net_tmp_1=None, img_tmp_1=None,net_tmp_2=None, img_tmp_2=None):#前传就是生成图片
       
        gen_imgs_1 = net_tmp_1(img_tmp_1)
        gen_imgs_2 = net_tmp_2(img_tmp_2)



        self.rec_A = self.netG_B(gen_imgs_1)
        self.rec_B = self.netG_A(gen_imgs_2)
        
        self.ssim=SSIM().cuda()
        output = gen_imgs_1.data.cpu().clamp(-1, 1)
        output = output
        gt = img_tmp_2.data.cpu()
        self.ssim_fakeB = self.ssim(output, gt).cpu().squeeze().detach().numpy()
        self.psnr_fakeB = self.PSNR(output.data.cpu().numpy() * 255, gt.data.cpu().numpy() * 255)
       

        return gen_imgs_1, gen_imgs_2

    def backward_D_basic(self, netD, real, fake):
 
        # Real
        pred_real = netD(real)
        pred_fake = netD(fake.detach())

        loss_D_fake,loss_D_real = self.criterionD(pred_fake, pred_real)#lp改
        # Combined loss and calculate gradients
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
      
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
   
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        #fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)

    def backward_D(self):
        """Calculate GAN loss for discriminator D_B and D_A"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
    
    def backward_Gnoevo(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        '''
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = (self.criterionIdt(self.idt_A, self.real_B)+self.mseloss(self.fake_B,self.real_B)) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = (self.criterionIdt(self.idt_B, self.real_A)+self.mseloss(self.fake_A,self.real_A)) * lambda_A * lambda_idt
            # self.idt_C = self.netG_C(self.real_B)
            # self.loss_idt_C = self.criterionIdt(self.idt_C,self.real_B) * lambda_B * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGANnoevo(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGANnoevo(self.netD_B(self.fake_A), True)
        self.loss_content=self.mseloss(self.fake_B,self.real_B)

        # self.loss_G_C = self.criterionGAN(self.netD_C(self.D_noise),True)
        # self.loss_G_C = self.criterionGAN(self.netD_C(self.zengq), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # self.loss_cycle_C = self.criterionCycle(self.zengq, self.real_B) * lambda_B
        # combined loss and calculate gradients
        # self.loss_noise = self.mseloss(self.D_noise,self.real_B)
       
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A+self.loss_idt_B+self.loss_content
        '''
        #lambda_A = self.opt.lambda_A
        #lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = (self.criterionIdt(self.idt_A, self.real_B)+self.mseloss(self.fake_B,self.real_B)) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = (self.criterionIdt(self.idt_B, self.real_A)+self.mseloss(self.fake_A,self.real_A)) * lambda_A * lambda_idt
           
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        
 
        self.loss_G_A = self.criterionGANnoevo(self.netD_A(self.fake_B), True)
       
        self.loss_G_B = self.criterionGANnoevo(self.netD_B(self.fake_A), True)
        


        self.loss_content=self.mseloss(self.fake_B,self.real_B)
        self.loss_vggA = self.criterionVGGA(self.fake_B,self.real_B)
        self.loss_vggB = self.criterionVGGB(self.fake_A,self.real_A)
       
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A+self.loss_idt_B+self.loss_content+self.loss_vggA+self.loss_vggB

        self.loss_G.backward(retain_graph=True)

    def backward_G(self,criterionGAN):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
    
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = (self.criterionIdt(self.idt_A, self.real_B)+self.mseloss(self.fake_B,self.real_B)) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = (self.criterionIdt(self.idt_B, self.real_A)+self.mseloss(self.fake_A,self.real_A)) * lambda_A * lambda_idt
            # self.idt_C = self.netG_C(self.real_B)
            # self.loss_idt_C = self.criterionIdt(self.idt_C,self.real_B) * lambda_B * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

       
        self.loss_G_A_fake,self.loss_G_A_real = criterionGAN(self.netD_A(self.fake_B.detach()),self.netD_A(self.real_B))
        self.loss_G_B_fake,self.loss_G_B_real = criterionGAN(self.netD_B(self.fake_A.detach()), self.netD_B(self.real_A))
        self.loss_G_A = self.loss_G_A_fake + self.loss_G_A_real
        self.loss_G_B = self.loss_G_B_fake + self.loss_G_B_real


   
        self.loss_content=self.mseloss(self.fake_B,self.real_B)

      
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
     
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_vggA = self.criterionVGGA(self.fake_B,self.real_B)
        self.loss_vggB = self.criterionVGGB(self.fake_A,self.real_A)
        self.loss_G = 1.2*(1.1*self.loss_G_A + self.loss_G_B) + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A+self.loss_idt_B+self.loss_content+self.loss_vggA+self.loss_vggB
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        tag="no_evo"
        if(self.Calculate_n%5==0):
            self.Evo_G()
            tag = "evo"
        else:
            self.Evo_G_no_evolution()
            
        self.set_requires_grad([self.netD_A, self.netD_B], True) 
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()
        self.optimizer_D.step()    # update D_A and D_B's weights


      
        self.Calculate_n = self.Calculate_n + 1
           
    # 进化函数
    def Evo_G(self):
        # eval_imgs = self.input_imgs[-self.eval_size:,:,:,:]
        # eval_targets = self.input_target[-self.eval_size:,:] 
        # define real images pass D


        F_A_list = np.zeros(self.opt.candi_num) 
        F_B_list = np.zeros(self.opt.candi_num)#两个大小为 self.opt.candi_num 的 NumPy 数组 
        # G_A = [0] G_B = [1]
        G_list = [[],[]]
        optG_list = []
        count = 0
       
        for i in range(self.opt.candi_num):
            self.G_candis[0][i]=self.netG_A.state_dict()
            self.G_candis[1][i]=self.netG_B.state_dict()
            self.optG_candis[i]=self.optimizer_G.state_dict()
        
        # 从候选集中选出最优的个体
        for i in range(self.opt.candi_num):#这个参数不应是candi_num！！！！！！！！！lp
            for j, criterionG in enumerate(self.G_mutations):  
                # Variation 
                with torch.autograd.set_detect_anomaly(True):#这句话是在 PyTorch 中用于启用异常检测（Anomaly Detection）的语句。 
                  
                    self.netG_A.load_state_dict(self.G_candis[0][i])
                    self.netG_B.load_state_dict(self.G_candis[1][i])
                    self.optimizer_G.load_state_dict(self.optG_candis[i])
                   
                    self.optimizer_G.zero_grad()
                    self.fake_B,self.fake_A= self.forward(self.netG_A,self.real_A,self.netG_B,self.real_B)
                    self.set_requires_grad([self.netD_A, self.netD_B], False)
                    self.backward_G(criterionG)
                    self.optimizer_G.step()

                # Evaluation 
                with torch.no_grad():
                    fake_B_evo,fake_A_evo= self.forward(self.netG_A,self.real_A,self.netG_B,self.real_B)
                #fake_B_evo,fake_A_evo = self.fake_B,self.fake_A
                #with torch.no_grad(): 
                   # fake_B
                F_G_A, F_G_B = self.fitness_score(fake_B_evo,fake_A_evo)
                
               
                # Selection 
                if count < self.opt.candi_num:
                    F_A_list[count] = F_G_A
                    F_B_list[count] = F_G_B
                    G_list[0].append(copy.deepcopy(self.netG_A.state_dict()))
                    G_list[1].append(copy.deepcopy(self.netG_B.state_dict()))
                    optG_list.append(copy.deepcopy(self.optimizer_G.state_dict()))
                    
                
                else:
                    fit_A_com = F_G_A - F_A_list
                    fit_B_com = F_G_B - F_B_list
                    if max(fit_A_com) > 0 and max(fit_B_com) > 0 :
                        ids_replace = np.where(fit_A_com==max(fit_A_com))[0][0]
                        F_A_list[ids_replace] = F_G_A
                        G_list[0][ids_replace] = copy.deepcopy(self.netG_A.state_dict())
                        optG_list[ids_replace] = copy.deepcopy(self.optimizer_G.state_dict())
                    
                        ids_replace = np.where(fit_B_com==max(fit_B_com))[0][0]
                        F_B_list[ids_replace] = F_G_B
                        G_list[1][ids_replace] = copy.deepcopy(self.netG_B.state_dict())
                        optG_list[ids_replace] = copy.deepcopy(self.optimizer_G.state_dict())
                      
                
                count += 1 
        self.G_candis = copy.deepcopy(G_list)             
        self.optG_candis = copy.deepcopy(optG_list)
        print('G_candis',len(self.G_candis[0]))
        
        #return np.array(Fit_list)
   
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
    # 适应度函数 评价函数
    def fitness_score(self,fake_B_evo,fake_A_evo):
        self.set_requires_grad([self.netD_A, self.netD_B], True)

        eval_fake = self.netD_A(fake_B_evo)  # D_A(G_A(A))
        eval_real = self.netD_A(self.real_B)  # D_A(B)
        self.ssim=SSIM().cuda()
       
        self.ssim_fakeB = self.ssim(fake_B_evo,self.real_B).cpu().squeeze().detach().numpy()
     
       

    
        eval_D_fake, eval_D_real = self.criterionD(eval_fake, eval_real) 
        eval_D = eval_D_fake + eval_D_real
        self.ssim_fakeA = self.ssim(fake_A_evo,self.real_A).cpu().squeeze().detach().numpy()

        F1 = 0 
        Fd = 0
      
        F1 += nn.Sigmoid()(eval_fake).data.mean().cpu().numpy()
        gradients = torch.autograd.grad(outputs=eval_D, inputs=self.netD_A.parameters(),
                        grad_outputs=torch.ones(eval_D.size()).to(self.device),
                        create_graph=True, retain_graph=True, only_inputs=True)
        with torch.no_grad():
            for j, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if j == 0 else torch.cat([allgrad,grad]) 
        Fd += -torch.log(torch.norm(allgrad)).data.cpu().numpy()
        fake_B_evo1 = util.tensor2im(fake_B_evo)
        Fuciqe =  self.getUCIQE(fake_B_evo1)
       
        print('FAq:',F1,'FAd:',self.opt.lambda_f*Fd,'Fssim:',self.ssim_fakeB,'Fuciqe:',Fuciqe)
      

        F_A_G = 1*F1 + self.opt.lambda_f * Fd + self.ssim_fakeB*0.5 + Fuciqe*0.1
        
      
        eval_fake = self.netD_B(fake_A_evo) 
        eval_real = self.netD_B(self.real_A)  
      
        F2 = nn.Sigmoid()(eval_fake).data.mean().cpu().numpy()
       
       
        eval_D_fake, eval_D_real = self.criterionD(eval_fake, eval_real) 
        eval_D = eval_D_fake + eval_D_real
        gradients = torch.autograd.grad(outputs=eval_D, inputs=self.netD_B.parameters(),
                                        grad_outputs=torch.ones(eval_D.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad,grad]) 
        Fd = -torch.log(torch.norm(allgrad)).data.cpu().numpy()
        
        print('FBq:',F2,'FBd:',self.opt.lambda_f*Fd,'Fssim:',self.ssim_fakeA)
        
        F_G_B = 1*F2 + self.opt.lambda_f * Fd + self.ssim_fakeA

        return F_A_G, F_G_B 
    


    def Evo_G_no_evolution(self):
        self.netG_A.load_state_dict(self.G_candis[0][0])
        self.netG_B.load_state_dict(self.G_candis[1][0])
        self.optimizer_G.load_state_dict(self.optG_candis[0])
        self.optimizer_G.zero_grad()
        self.fake_B,self.fake_A= self.forward(self.netG_A,self.real_A,self.netG_B,self.real_B)
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.backward_G(self.G_mutations[1])
        self.optimizer_G.step()
