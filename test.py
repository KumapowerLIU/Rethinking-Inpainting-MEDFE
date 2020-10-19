import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
#from torch.utils.tensorboard import SummaryWriter
import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
if __name__ == "__main__":

    def img2tensor(pimg):
        img = np.array(pimg)
        img = img/255.0
        img = torch.Tensor(img.transpose((2, 0, 1)))[None, ...]
        img = (img-0.5)/0.5
        return img
    def msk2tensor(pimg):
        img = np.array(pimg)
        img = (img>0)
        img = torch.Tensor(img)[None, None, ...].float()
        return img


    opt = TestOptions().parse()
    model = create_model(opt)
    model.netEN.module.load_state_dict(torch.load("EN.pkl"))
    model.netDE.module.load_state_dict(torch.load("DE.pkl"))
    model.netMEDFE.module.load_state_dict(torch.load("MEDEF.pkl"))
    for name in tqdm(os.listdir(opt.mask_root)):
        path_m = f"{opt.mask_root}/{name}"
        path_d = f"{opt.de_root}/{name}"
        path_s = f"{opt.st_root}/{name}"
        mask = Image.open(path_m).convert("L")
        detail = Image.open(path_d).convert("RGB")
        structure = Image.open(path_s).convert("RGB")

        mask = msk2tensor(mask)
        detail = img2tensor(detail)
        structure = img2tensor(structure)
        with torch.no_grad():
            model.set_input(detail, structure, mask)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * mask + detail*(1-mask)
            fake_image = (fake_out+1)/2.0
        output = fake_image.detach().numpy()[0].transpose((1,2,0))*255
        output = Image.fromarray(output.astype(np.uint8))
        output.save(f"{opt.results_dir}/{name}")
