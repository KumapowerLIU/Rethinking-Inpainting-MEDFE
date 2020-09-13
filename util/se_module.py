from torch import nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1,stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c,1,1)
        y = self.fc(y)
        return x * y.expand_as(x)
        # b, c, _, _ = x.size()
        # latter=x.clone()
        # y = self.avg_pool(x).view(b, c,1,1)
        #
        # y = self.fc(y)
        # top,ind=torch.topk(y,int(c/2),1)
        # ind=ind.view(-1)
        #
        # x=torch.index_select(x,1,ind)
        #
        # return torch.cat([x, latter], 1)
