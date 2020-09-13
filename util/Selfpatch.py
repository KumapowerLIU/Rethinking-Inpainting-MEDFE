import torch
import torch.nn as nn


class Selfpatch(object):
    def buildAutoencoder(self, target_img, target_img_2, target_img_3, patch_size=1, stride=1):
        nDim = 3
        assert target_img.dim() == nDim, 'target image must be of dimension 3.'
        C = target_img.size(0)

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor

        patches_features = self._extract_patches(target_img, patch_size, stride)
        patches_features_f = self._extract_patches(target_img_3, patch_size, stride)

        patches_on = self._extract_patches(target_img_2, 1, stride)

        return patches_features_f, patches_features, patches_on

    def build(self, target_img,  patch_size=5, stride=1):
        nDim = 3
        assert target_img.dim() == nDim, 'target image must be of dimension 3.'
        C = target_img.size(0)

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor

        patches_features = self._extract_patches(target_img, patch_size, stride)

        return patches_features

    def _build(self, patch_size, stride, C, target_patches, npatches, normalize, interpolate, type):
        # for each patch, divide by its L2 norm.
        if type == 1:
            enc_patches = target_patches.clone()
            for i in range(npatches):
                enc_patches[i] = enc_patches[i]*(1/(enc_patches[i].norm(2)+1e-8))

            conv_enc = nn.Conv2d(npatches, npatches, kernel_size=1, stride=stride, bias=False, groups=npatches)
            conv_enc.weight.data = enc_patches
            return conv_enc

        # normalize is not needed, it doesn't change the result!
            if normalize:
                raise NotImplementedError

            if interpolate:
                raise NotImplementedError
        else:

            conv_dec = nn.ConvTranspose2d(npatches, C, kernel_size=patch_size, stride=stride, bias=False)
            conv_dec.weight.data = target_patches
            return conv_dec

    def _extract_patches(self, img, patch_size, stride):
        n_dim = 3
        assert img.dim() == n_dim, 'image must be of dimension 3.'
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        i_1, i_2, i_3, i_4, i_5 = input_windows.size(0), input_windows.size(1), input_windows.size(2), input_windows.size(3), input_windows.size(4)
        input_windows = input_windows.permute(1,2,0,3,4).contiguous().view(i_2*i_3, i_1, i_4, i_5)
        patches_all = input_windows
        return patches_all



