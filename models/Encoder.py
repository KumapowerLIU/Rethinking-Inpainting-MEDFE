import torch.nn as nn


# Define the resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# define the Encoder unit
class UnetSkipConnectionEBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionEBlock, self).__init__()
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)

        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = norm_layer(inner_nc, affine=True)
        if outermost:
            down = [downconv]
            model = down
        elif innermost:
            down = [downrelu, downconv]
            model = down
        else:
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                model = down + [nn.Dropout(0.5)]
            else:
                model = down
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, res_num=4, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Encoder, self).__init__()

        # construct unet structure
        Encoder_1 = UnetSkipConnectionEBlock(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, outermost=True)
        Encoder_2 = UnetSkipConnectionEBlock(ngf, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout)
        Encoder_3 = UnetSkipConnectionEBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout)
        Encoder_4 = UnetSkipConnectionEBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        Encoder_5 = UnetSkipConnectionEBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        Encoder_6 = UnetSkipConnectionEBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout, innermost=True)

        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.Encoder_1 = Encoder_1
        self.Encoder_2 = Encoder_2
        self.Encoder_3 = Encoder_3
        self.Encoder_4 = Encoder_4
        self.Encoder_5 = Encoder_5
        self.Encoder_6 = Encoder_6

    def forward(self, input):
        y_1 = self.Encoder_1(input)
        y_2 = self.Encoder_2(y_1)
        y_3 = self.Encoder_3(y_2)
        y_4 = self.Encoder_4(y_3)
        y_5 = self.Encoder_5(y_4)
        y_6 = self.Encoder_6(y_5)
        y_7 = self.middle(y_6)

        return y_1, y_2, y_3, y_4, y_5, y_7
