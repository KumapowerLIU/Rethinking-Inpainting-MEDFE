import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import util.util as util
class InnerCos(nn.Module):
    def __init__(self):
        super(InnerCos, self).__init__()
        self.criterion = nn.L1Loss()
        self.target = None
        self.down_model = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1,stride=1, padding=0),
            nn.Tanh()
        )

    def set_target(self, targetde, targetst):
        self.targetst = F.interpolate(targetst, size=(32, 32), mode='bilinear')
        self.targetde = F.interpolate(targetde, size=(32, 32), mode='bilinear')

    def get_target(self):
        return self.target

    def forward(self, in_data):
        loss_co = in_data[1]
        self.ST = self.down_model(loss_co[0])
        self.DE = self.down_model(loss_co[1])
        self.loss = self.criterion(self.ST, self.targetst)+self.criterion(self.DE, self.targetde)
        self.output = in_data[0]
        return self.output

    def backward(self, retain_graph=True):

        self.loss.backward(retain_graph=retain_graph)
        return self.loss

    def __repr__(self):

        return self.__class__.__name__