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
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
if __name__ == "__main__":

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


    opt = TestOptions().parse()
    model = create_model(opt)
    model.netEN.module.load_state_dict(torch.load("EN.pkl"))
    model.netDE.module.load_state_dict(torch.load("DE.pkl"))
    model.netMEDFE.module.load_state_dict(torch.load("MEDEF.pkl"))
    results_dir = r'./result'
    if not os.path.exists( results_dir):
        os.mkdir(results_dir)

    mask_paths = glob('{:s}/*'.format(opt.mask_root))
    de_paths = glob('{:s}/*'.format(opt.de_root))
    st_path = glob('{:s}/*'.format(opt.st_root))
    image_len = len(de_paths )
    for i in tqdm(range(image_len)):
        # only use one mask for all image
        path_m = mask_paths[0]
        path_d = de_paths[i]
        path_s = de_paths[i]

        mask = Image.open(path_m).convert("RGB")
        detail = Image.open(path_d).convert("RGB")
        structure = Image.open(path_s).convert("RGB")


        mask = mask_transform(mask)
        detail = img_transform(detail)
        structure = img_transform(structure)
        mask = torch.unsqueeze(mask, 0)
        detail = torch.unsqueeze(detail, 0)
        structure = torch.unsqueeze(structure,0)

        with torch.no_grad():
            model.set_input(detail, structure, mask)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * mask + detail*(1-mask)
            fake_image = (fake_out+1)/2.0
        output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
        output = Image.fromarray(output.astype(np.uint8))
        output.save(rf"{opt.results_dir}/{i}.png")
