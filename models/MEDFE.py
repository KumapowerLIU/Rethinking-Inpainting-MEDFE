import torch
import random
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from PIL import Image
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .loss import VGG16, PerceptualLoss, StyleLoss, GANLoss


class MEDFE(BaseModel):
    def __init__(self, opt):
        super(MEDFE, self).__init__(opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        self.device = torch.device('cuda')
        # define tensors
        self.vgg = VGG16()
        self.PerceptualLoss = PerceptualLoss()
        self.StyleLoss = StyleLoss()
        self.input_DE = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_ST = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.Gt_Local = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.Gt_DE = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.Gt_ST = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.mask_global = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        self.model_names = []
        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.vgg = self.vgg.to(self.gpu_ids[0])
            self.vgg = torch.nn.DataParallel(self.vgg, self.gpu_ids)
        # load/define networks  EN:Encoder DE:Decoder  MEDFE: Mutual Encoder Decoder with Feature Equalizations
        self.netEN, self.netDE, self.netMEDFE, self.stde_loss = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm,
                                                                                  opt.use_dropout, opt.init_type,
                                                                                  self.gpu_ids,
                                                                                  opt.init_gain)
        self.model_names=['EN', 'DE', 'MEDFE']


        if self.isTrain:

            self.netD = networks.define_D(3, opt.ndf,
                                          opt.n_layers_D, opt.norm, opt.init_type, self.gpu_ids, opt.init_gain)
            self.netF = networks.define_D(3, opt.ndf,
                                          opt.n_layers_D, opt.norm, opt.init_type, self.gpu_ids, opt.init_gain)
            self.model_names.append('D')
            self.model_names.append('F')
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = GANLoss(tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_EN = torch.optim.Adam(self.netEN.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DE = torch.optim.Adam(self.netDE.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_MEDFE = torch.optim.Adam(self.netMEDFE.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_EN)
            self.optimizers.append(self.optimizer_DE)

            self.optimizers.append(self.optimizer_MEDFE)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netEN)
            networks.print_network(self.netDE)
            networks.print_network(self.netMEDFE)
            if self.isTrain:
                networks.print_network(self.netD)
                networks.print_network(self.netF)
            print('-----------------------------------------------')
            #####modified
        if self.isTrain:
            if opt.continue_train :
                print('Loading pre-trained network!')
                self.load_networks(self.netEN, 'EN', opt.which_epoch)
                self.load_networks(self.netDE, 'DE', opt.which_epoch)
                self.load_networks(self.netMEDFE, 'MEDFE', opt.which_epoch)
                self.load_networks(self.netD, 'D', opt.which_epoch)
                self.load_networks(self.netF, 'F', opt.which_epoch)

    def name(self):
        return self.modlename

    def mask_process(self, mask):
        mask = mask[0][0]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 1)
        mask = mask.byte()
        return mask


    def set_input(self, input_De, input_St, mask):

        self.Gt_DE = input_De.to(self.device)
        self.Gt_ST = input_St.to(self.device)
        self.input_DE = input_De.to(self.device)
        self.mask_global = self.mask_process(mask.to(self.device))
        self.Gt_Local = input_De.to(self.device)
        # define local area which send to the local discriminator
        self.crop_x = random.randint(0, 191)
        self.crop_y = random.randint(0, 191)
        self.Gt_Local = self.Gt_Local[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        self.ex_mask = self.mask_global.expand(self.mask_global.size(0), 3, self.mask_global.size(2),
                                               self.mask_global.size(3))
        #  unpositve with original mask
        self.inv_ex_mask = torch.add(torch.neg(self.ex_mask.float()), 1).float()
        # set loss groundtruth for two branch
        self.stde_loss[0].set_target(self.Gt_DE, self.Gt_ST)
        # Do not set the mask regions as 0
        self.input_DE.narrow(1, 0, 1).masked_fill_(self.mask_global.narrow(1, 0, 1).bool(), 2 * 123.0 / 255.0 - 1.0)
        self.input_DE.narrow(1, 1, 1).masked_fill_(self.mask_global.narrow(1, 0, 1).bool(), 2 * 104.0 / 255.0 - 1.0)
        self.input_DE.narrow(1, 2, 1).masked_fill_(self.mask_global.narrow(1, 0, 1).bool(), 2 * 117.0 / 255.0 - 1.0)

    def forward(self):

        fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6 = self.netEN(
            torch.cat([self.input_DE, self.inv_ex_mask], 1))
        De_in = [fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6]
        x_out = self.netMEDFE(De_in, self.mask_global)
        self.fake_out = self.netDE(x_out[0], x_out[1], x_out[2], x_out[3], x_out[4], x_out[5])

    def backward_D(self):
        fake_AB = self.fake_out
        real_AB = self.Gt_DE  # GroundTruth
        real_local = self.Gt_Local
        fake_local = self.fake_out[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global Discriminator
        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)

        # Local discriminator
        self.pred_fake_F = self.netF(fake_local.detach())
        self.pred_real_F = self.netF(real_local)
        self.loss_F_fake = self.criterionGAN(self.pred_fake_F, self.pred_real_F, True)

        self.loss_D = self.loss_D_fake + self.loss_F_fake
        self.loss_D.backward()

    def backward_G(self):
        # First, The generator should fake the discriminator
        real_AB = self.Gt_DE
        fake_AB = self.fake_out
        real_local = self.Gt_Local
        fake_local = self.fake_out[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global discriminator
        pred_real = self.netD(real_AB)
        pred_fake = self.netD(fake_AB)
        # Local discriminator
        pred_real_F = self.netF(real_local)
        pred_fake_f = self.netF(fake_local)
        self.loss_G_GAN = self.criterionGAN(pred_fake, pred_real, False) + self.criterionGAN(pred_fake_f, pred_real_F,
                                                                                             False)
        # Second, Reconstruction loss
        self.loss_L1 = self.criterionL1(self.fake_out, self.Gt_DE)
        self.Perceptual_loss = self.PerceptualLoss(self.fake_out, self.Gt_DE)
        self.Style_Loss = self.StyleLoss(self.fake_out, self.Gt_DE)

        # self.loss_G = self.loss_G_L1 + self.loss_G_GAN *0.2 + self.Perceptual_loss * 0.2 + self.Style_Loss *250
        self.loss_G = self.loss_L1 * self.opt.lambda_L1 + self.loss_G_GAN * self.opt.lambda_Gan + \
                      self.Perceptual_loss * self.opt.lambda_P + self.Style_Loss * self.opt.lambda_S

        self.stde_loss_value = 0
        for loss in self.stde_loss:
            self.stde_loss_value += loss.backward()
            self.stde_loss_value += loss.loss
        self.loss_G += self.stde_loss_value
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # Optimize the D and F first
        self.set_requires_grad(self.netF, True)
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netEN, False)
        self.set_requires_grad(self.netDE, False)
        self.set_requires_grad(self.netMEDFE, False)
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_F.step()

        # Optimize EN, DE, MEDEF
        self.set_requires_grad(self.netF, False)
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netEN, True)
        self.set_requires_grad(self.netDE, True)
        self.set_requires_grad(self.netMEDFE, True)
        self.optimizer_EN.zero_grad()
        self.optimizer_DE.zero_grad()
        self.optimizer_MEDFE.zero_grad()
        self.backward_G()
        self.optimizer_MEDFE.step()
        self.optimizer_EN.step()
        self.optimizer_DE.step()

    def get_current_errors(self):
        # show the current loss
        return OrderedDict([('G_GAN', self.loss_G_GAN.data),
                            ('G_L1', self.loss_G.data),
                            ('G_stde', self.stde_loss_value.data),
                            ('D', self.loss_D_fake.data),
                            ('F', self.loss_F_fake.data)
                            ])

    # You can also see the Tensorborad
    def get_current_visuals(self):
        input_image = (self.input_DE.data.cpu()+1)/2.0
        fake_image = (self.fake_out.data.cpu()+1)/2.0
        real_gt = (self.Gt_DE.data.cpu()+1)/2.0
        return input_image, fake_image, real_gt
    
