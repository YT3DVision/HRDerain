import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from thop import profile

try:
    # from mmcv.ops.carafe import normal_init, xavier_init, carafe
    from mmcv.ops import car
except ImportError:

    def carafe(x, normed_mask, kernel_size, group = 1, up = 1):
        b, c, h, w = x.shape
        _, m_c, m_h, m_w = normed_mask.shape

        pad = kernel_size // 2
        pad_x = F.pad(x, pad=[pad] * 4, mode='reflect') # pad_x torch.Size([1, 3, 962, 542])

        unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0) # unfold_x 1 torch.Size([1, 27, 518400])
        unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w) # unfold_x 2 torch.Size([1, 27, 960, 540])
        unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest') # unfold_x 3 torch.Size([1, 27, 1920, 1080])
        unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w) # unfold_x 4 torch.Size([1, 3, 9, 1920, 1080])

        normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w) # normed_mask torch.Size([1, 1, 9, 1920, 1080])

        res = unfold_x * normed_mask # res 1 torch.Size([1, 3, 9, 1920, 1080])
        res = res.sum(dim=2).reshape(b, c, m_h, m_w) # res 2 torch.Size([1, 3, 1920, 1080])

        return res

    def xavier_init(module: nn.Module,
                    gain: float = 1,
                    bias: float = 0,
                    distribution: str = 'normal') -> None:
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)





def xxx(x, normed_mask, kernel_size = 3, group = 1, up =1):
    b, c, _, _ = x.shape
    _, _, m_h, m_w = normed_mask.shape

    x= F.interpolate(x, scale_factor=up, mode='nearest')
    x = x.reshape(b, c, 1, m_h, m_w)
    normed_mask = normed_mask.reshape(1, 1, kernel_size * kernel_size, m_h, m_w)
    res = x * normed_mask
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)

    return res


class simplefreqfusion(nn.Module):
    def __init__(self,
                hr_channels,
                lr_channels,
                scale_factor=1,
                lowpass_kernel=5,
                highpass_kernel=3,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64,
                align_corners=False,
                upsample_mode='nearest',
                comp_feat_upsample=True, # use ALPF & AHPF for init upsampling
                use_high_pass=True,
                hr_residual=True,
                semi_conv=True,
                ):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels,1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels,1)
        self.content_encoder = nn.Conv2d( # ALPF generator
            self.compressed_channels,
            lowpass_kernel ** 2 * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.semi_conv = semi_conv
        self.comp_feat_upsample = comp_feat_upsample
        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d( # AHPF generator
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
            
        self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
        self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel**2)) # group
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        # print(hamming)
        # print(mask.shape)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask =  mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        # printGPU()
        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_high_pass:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat) #从hr_feat得到初始高通滤波特征
                    mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel, hamming=self.hamming_highpass) #kernel归一化得到初始高通滤波
                    # print("1")
                    # printGPU()
                    compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat, mask_hr_init, self.highpass_kernel, self.up_group, 1) #利用初始高通滤波对压缩hr_feat的高频增强 （x-x的低通结果=x的高通结果）
                    # print("2")
                    # printGPU()
                    mask_lr_hr_feat = self.content_encoder(compressed_hr_feat) #从hr_feat得到初始低通滤波特征
                    mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, hamming=self.hamming_lowpass) #kernel归一化得到初始低通滤波
                    # print("3")
                    # printGPU()
                    mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat) #从hr_feat得到另一部分初始低通滤波特征
                    mask_lr_lr_feat = F.interpolate( #利用初始低通滤波对另一部分初始低通滤波特征上采样
                        carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat #将两部分初始低通滤波特征合在一起
                    # mask_lr = mask_lr_lr_feat #########################################################################################

                    # mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass) #得到初步融合的初始低通滤波
                    # mask_hr_lr_feat = F.interpolate( #使用初始低通滤波对lr_feat处理，分辨率得到提高
                    #     carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    # mask_hr = mask_hr_hr_feat + mask_hr_lr_feat # 最终高通滤波特征
                    mask_hr = mask_hr_hr_feat #########################################################################################
                    # print("4")
                    # printGPU()
                    
                else: raise NotImplementedError
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
        else:
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:], mode='nearest') + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_high_pass: 
                mask_hr = self.content_encoder2(compressed_x)
        ###
        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
        if self.semi_conv:
                lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)
        else:
            lr_feat = resize(
                input=lr_feat,
                size=hr_feat.shape[2:],
                mode=self.upsample_mode,
                align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 1)
        # print("5")
        # printGPU()

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
            hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
            if self.hr_residual:
                hr_feat = hr_feat_hf + hr_feat
            else:
                hr_feat = hr_feat_hf
        # print("6")
        # printGPU()
        return  mask_lr, hr_feat, lr_feat




from pynvml import nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, nvmlShutdown

def printGPU():
    nvmlInit()
    total_memory = 0
    total_free = 0
    total_used = 0
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    gpu_name = nvmlDeviceGetName(handle)
    print("GPU: {}".format(gpu_name), end="    ")
    print("总共显存: {}G".format((info.total // 1048576) / 1024), end="    ")
    print("空余显存: {}G".format((info.free // 1048576) / 1024), end="    ")
    print("已用显存: {}G".format((info.used // 1048576) / 1024), end="    ")
    print("显存占用率: {:.2%}".format( info.used / info.total))
    nvmlShutdown()







# net
class  old_version_fus(nn.Module):
    def __init__(self, hr_channels, lr_channels, lowpass_kernel=5, highpass_kernel=3, compressed_channels=64):
        super().__init__()
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.compressed_channels = compressed_channels
        self.encoder_kernel = 3
        self.encoder_dilation = 1
        self.up_group = 1
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)

        self.content_encoder = nn.Conv2d( # ALPF generator
            self.compressed_channels,
            self.lowpass_kernel ** 2 * self.up_group ,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.content_encoder2 = nn.Conv2d( # AHPF generator
            self.compressed_channels,
            self.highpass_kernel ** 2 * self.up_group ,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
        self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        # if self.use_high_pass:
        normal_init(self.content_encoder2, std=0.001)
    
    #在应用于输入特征图之前，每个 kup × kup 重组核在空间上用 softmax 函数进行归一化。归一化步骤强制核值之和为 1，这是一个跨局部区域的软选择。
    # 由于内核归一化器，CARAFE 不执行任何 重新缩放 和 更改特征图的平均值，这就是为什么我们提出的运算符被命名为特征重组。
    def kernel_normalizer(self, mask, kernel, scale_factor = None, hamming = 1):
        if scale_factor is not None:
            # mask = F.pixel_unshuffle(mask, scale_factor)
            mask = F.interpolate(mask, scale_factor=scale_factor, mode='nearest', recompute_scale_factor=True)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel**2)) # group
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask =  mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

        
    def forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        # printGPU()

        # low freq
        mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
        mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, scale_factor = 0.5, hamming=self.hamming_lowpass)
        mask_lr = F.interpolate(mask_lr_hr_feat, size=compressed_lr_feat.shape[-2:], mode='nearest') + carafe(self.content_encoder(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 1)
        # mask_lr = mask_lr_hr_feat(down) + mask_lr_lr_feat
        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)

        lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 1) + lr_feat
        # printGPU()

        # high freq
        # mask_hr_lr_feat = F.interpolate(carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 1), size=compressed_hr_feat.shape[-2:], mode='nearest')
        mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 1), size=compressed_hr_feat.shape[-2:], mode='nearest')
        mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
        # mask_hr = mask_hr_hr_feat + mask_hr_lr_feat(up)
        # printGPU()
        hr_feat = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1) + hr_feat

        return  hr_feat, lr_feat



class  fus(nn.Module):
    def __init__(self, hr_channels, lr_channels, lowpass_kernel=5, highpass_kernel=3, compressed_channels=64, scale_factor = 0.5):
        super().__init__()
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.compressed_channels = compressed_channels
        self.encoder_kernel = 3
        self.encoder_dilation = 1
        self.up_group = 1
        self.alian_mode = 'bilinear'
        self.scale_factor = scale_factor
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)

        self.content_encoder = nn.Conv2d( # ALPF generator
            self.compressed_channels,
            self.lowpass_kernel ** 2 * self.up_group ,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.content_encoder2 = nn.Conv2d( # AHPF generator
            self.compressed_channels,
            self.highpass_kernel ** 2 * self.up_group ,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
        self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        # if self.use_high_pass:
        normal_init(self.content_encoder2, std=0.001)
    
    #在应用于输入特征图之前，每个 kup × kup 重组核在空间上用 softmax 函数进行归一化。归一化步骤强制核值之和为 1，这是一个跨局部区域的软选择。
    # 由于内核归一化器，CARAFE 不执行任何 重新缩放 和 更改特征图的平均值，这就是为什么我们提出的运算符被命名为特征重组。
    def kernel_normalizer(self, mask, kernel, scale_factor = None, hamming = 1):
        if scale_factor is not None:
            # mask = F.pixel_unshuffle(mask, scale_factor)
            mask = F.interpolate(mask, scale_factor=scale_factor, mode=self.alian_mode, recompute_scale_factor=True)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel**2)) # group
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask =  mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

        
    def forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        # printGPU()

        # low freq
        mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
        mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, scale_factor = self.scale_factor, hamming=self.hamming_lowpass)
        mask_lr = F.interpolate(mask_lr_hr_feat, size=compressed_lr_feat.shape[-2:], mode=self.alian_mode) + carafe(self.content_encoder(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 1)
        # mask_lr = mask_lr_hr_feat(down) + mask_lr_lr_feat
        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)

        lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 1) + lr_feat
        # printGPU()

        # high freq
        # mask_hr_lr_feat = F.interpolate(carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 1), size=compressed_hr_feat.shape[-2:], mode=self.alian_mode)
        mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(carafe(self.content_encoder2(compressed_lr_feat), mask_lr, self.lowpass_kernel, self.up_group, 1), size=compressed_hr_feat.shape[-2:], mode=self.alian_mode)
        mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
        # mask_hr = mask_hr_hr_feat + mask_hr_lr_feat(up)
        # printGPU()
        hr_feat = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1) + hr_feat

        return  hr_feat, lr_feat



# net7
class  new_fus(nn.Module):
    def __init__(self, hr_channels, lr_channels, lowpass_kernel=5, highpass_kernel=3, compressed_channels=64):
        super().__init__()
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.compressed_channels = compressed_channels
        self.encoder_kernel = 3
        self.encoder_dilation = 1
        self.up_group = 1
        self.alian_mode = 'bilinear'
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)

        self.content_encoder = nn.Conv2d( # ALPF generator
            self.compressed_channels,
            self.lowpass_kernel ** 2 * self.up_group ,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.content_encoder2 = nn.Conv2d( # AHPF generator
            self.compressed_channels,
            self.highpass_kernel ** 2 * self.up_group ,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
        self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        # if self.use_high_pass:
        normal_init(self.content_encoder2, std=0.001)
    
    #在应用于输入特征图之前，每个 kup × kup 重组核在空间上用 softmax 函数进行归一化。归一化步骤强制核值之和为 1，这是一个跨局部区域的软选择。
    # 由于内核归一化器，CARAFE 不执行任何 重新缩放 和 更改特征图的平均值，这就是为什么我们提出的运算符被命名为特征重组。
    def kernel_normalizer(self, mask, kernel, scale_factor = None, hamming = 1):
        if scale_factor is not None:
            # mask = F.pixel_unshuffle(mask, scale_factor)
            mask = F.interpolate(mask, scale_factor=scale_factor, mode=self.alian_mode, recompute_scale_factor=True)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel**2)) # group
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask =  mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

        
    def forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        # printGPU()

        # low freq
        mask_lr_hr_feat = F.interpolate(self.content_encoder(compressed_hr_feat), size=compressed_lr_feat.shape[-2:], mode=self.alian_mode)
        mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, hamming=self.hamming_lowpass)
        mask_lr = mask_lr_hr_feat + carafe(self.content_encoder(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 1)
        # mask_lr = mask_lr_hr_feat(down) + mask_lr_lr_feat
        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)

        lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 1) + lr_feat
        # printGPU()

        # high freq
        mask_hr_lr_feat = F.interpolate(carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 1), size=compressed_hr_feat.shape[-2:], mode=self.alian_mode)
        mask_hr = self.content_encoder2(compressed_hr_feat) + mask_hr_lr_feat # F.interpolate(carafe(self.content_encoder2(compressed_lr_feat), mask_lr, self.lowpass_kernel, self.up_group, 1), size=compressed_hr_feat.shape[-2:], mode=self.alian_mode)
        mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
        # mask_hr = mask_hr_hr_feat + mask_hr_lr_feat(up)
        # printGPU()
        hr_feat = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1) + hr_feat

        return  hr_feat, lr_feat



if __name__ == '__main__':
    # printGPU()
    # print("start")
    # c = 32

    # c1 = int(2 * c)
    # c2 = int(c1 / 2)
    # # H = int(3840/4) # 480
    # # W = int(2160/4) # 270
    # H = 960
    # W = 540
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # hr_feat = torch.rand(1, c1, H, W).to(device)
    # lr_feat = torch.rand(1, c2, int(H/2), int(W/2)).to(device)
    # model = new_fus(hr_channels=c1, lr_channels=c2, compressed_channels=32).to(device)
    # # # Fu = SKFusion(3, height=2, reduction=8).to(device)
    # flops, params = profile(model, inputs=(hr_feat,lr_feat))
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))

    # print("end")
    # printGPU()
    # # print("hr_out",hr_feat.shape)
    # # print("lr_out",lr_feat.shape)
    # # # out = torch.cat([hr_feat, lr_feat], dim=1)
    # # out = Fu([hr_feat, lr_feat])
    # # print("out",out.shape)


    # # mask = torch.rand(1, c, H, W).to(device)
    # # # out_feat = xxx(x = lr_feat, normed_mask = mask, kernel_size = 3, up=2)
    # # out_feat = hr_feat*mask
    # # printGPU()
    # # print(out_feat.shape)


    printGPU()
    print("start")
    c = 32

    c1 = int(2 * c)
    c2 = int(c1 / 2)
    # H = int(3840/4) # 480
    # W = int(2160/4) # 270
    H = 960
    W = 540
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hr_feat = torch.rand(1, c1, H, W).to(device)
    lr_feat = torch.rand(1, c2, int(H/4), int(W/4)).to(device)
    model = new_fus(hr_channels=c1, lr_channels=c2, compressed_channels=32).to(device)
    # # Fu = SKFusion(3, height=2, reduction=8).to(device)
    flops, params = profile(model, inputs=(hr_feat,lr_feat))
    print(' Number of parameters:%.4f M' % (params / 1e6))
    print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))

    print("end")
    printGPU()










"""
    c = 32

    c1 = int(2 * c)
    c2 = int(c1 / 2)
    H = int(3840/2) # 480
    W = int(2160/2) # 270

new version
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 7.8984375G    已用显存: 0.1015625G    显存占用率: 1.27%
start
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 6.431640625G    已用显存: 1.5673828125G    显存占用率: 19.60%
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 5.509765625G    已用显存: 2.4892578125G    显存占用率: 31.13%
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 5.1796875G    已用显存: 2.8193359375G    显存占用率: 35.25%
end
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 3.62109375G    已用显存: 4.3779296875G    显存占用率: 54.74%

new new version
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 7.8984375G    已用显存: 0.1015625G    显存占用率: 1.27%
start
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 6.431640625G    已用显存: 1.5673828125G    显存占用率: 19.60%
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 4.927734375G    已用显存: 3.0712890625G    显存占用率: 38.40%
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 4.5234375G    已用显存: 3.4755859375G    显存占用率: 43.46%
end
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 3.03515625G    已用显存: 4.9638671875G    显存占用率: 62.06%














new design
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 7.8984375G    已用显存: 0.1015625G    显存占用率: 1.27%
start
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 6.958984375G    已用显存: 1.0400390625G    显存占用率: 13.01%
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 6.728515625G    已用显存: 1.2705078125G    显存占用率: 15.89%
end
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 6.1484375G    已用显存: 1.8505859375G    显存占用率: 23.14%








ful conv
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 7.8984375G    已用显存: 0.1015625G    显存占用率: 1.27%
start
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 6.728515625G    已用显存: 1.2705078125G    显存占用率: 15.89%
end
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 4.353515625G    已用显存: 3.6455078125G    显存占用率: 45.58%



only fft
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 7.8984375G    已用显存: 0.1015625G    显存占用率: 1.27%
start
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 5.697265625G    已用显存: 2.3017578125G    显存占用率: 28.78%
end
GPU: GeForce GTX 1070    总共显存: 8.0G    空余显存: 4.71875G    已用显存: 3.2802734375G    显存占用率: 41.02%
"""