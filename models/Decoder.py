import torch.nn as nn
import torch

class UnetSkipConnectionDBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)
        upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                    kernel_size=4, stride=2,
                                    padding=1)
        up = [uprelu, upconv, upnorm]

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Decoder, self).__init__()

        # construct unet structure
        Decoder_1 = UnetSkipConnectionDBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout,
                                             innermost=True)
        Decoder_2 = UnetSkipConnectionDBlock(ngf * 16, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_3 = UnetSkipConnectionDBlock(ngf * 16, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_4 = UnetSkipConnectionDBlock(ngf * 8, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_5 = UnetSkipConnectionDBlock(ngf * 4, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_6 = UnetSkipConnectionDBlock(ngf * 2, output_nc, norm_layer=norm_layer, use_dropout=use_dropout, outermost=True)

        self.Decoder_1 = Decoder_1
        self.Decoder_2 = Decoder_2
        self.Decoder_3 = Decoder_3
        self.Decoder_4 = Decoder_4
        self.Decoder_5 = Decoder_5
        self.Decoder_6 = Decoder_6

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        y_1 = self.Decoder_1(input_6)
        y_2 = self.Decoder_2(torch.cat([y_1, input_5], 1))
        y_3 = self.Decoder_3(torch.cat([y_2, input_4], 1))
        y_4 = self.Decoder_4(torch.cat([y_3, input_3], 1))
        y_5 = self.Decoder_5(torch.cat([y_4, input_2], 1))
        y_6 = self.Decoder_6(torch.cat([y_5, input_1], 1))
        out = y_6

        return out
