##########################net 11_2        在net11的基础上使用特殊残差链接     using:DCSAF and DCTfusion############################################
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.NAFBlock import NAFBlock
    from models.restormer_arch import TransformerBlock, Upsample, Downsample
    from models.simFreq import fus as Mutual
    from models.AttentionFusion import DCSAF, DCTFusion
except:
    from NAFBlock import NAFBlock
    from restormer_arch import TransformerBlock, Upsample, Downsample
    from simFreq import fus as Mutual
    from AttentionFusion import DCSAF, DCTFusion

from einops import rearrange
from thop import profile
from pynvml import nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, nvmlShutdown
import matplotlib.pyplot as plt




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







c = 32
c2 = 32
class Encoder(nn.Module):
    def __init__(self,
                 dim_h = c,
                 dim_l = c2,
                 num_blocks_hr=[1, 1, 4, 8],
                 num_blocks_lr=[1, 2, 2, 2],
                 heads_en=[1, 2, 4, 8], 
                 ffn_e_f=2.66,
                 LN_type='WithBias',
                 bias=False,):
        super(Encoder, self).__init__()

        self.encoder_H0 = nn.Sequential(*[NAFBlock(c = dim_h * 2 ** 0)for i in range(num_blocks_hr[0])])
        self.down_h0_1 = Downsample(dim_h * 2 ** 0)

        self.encoder_H1 = nn.Sequential(*[NAFBlock(c = int(dim_h * 2 ** 1))for i in range(num_blocks_hr[1])])
        self.down_h1_2 = Downsample(int(dim_h * 2 ** 1))

        self.encoder_H2 = nn.Sequential(*[NAFBlock(c = int(dim_h * 2 ** 2))for i in range(num_blocks_hr[2])])
        self.down_h2_3 = Downsample(int(dim_h * 2 ** 2))

        self.encoder_H3 = nn.Sequential(*[NAFBlock(c = int(dim_h * 2 ** 3))for i in range(num_blocks_hr[3])])

        ###########################################################
        self.encoder_L0 = nn.Sequential(*[TransformerBlock(dim=dim_l * 2 ** 0, num_heads=heads_en[0], ffn_expansion_factor=ffn_e_f, 
                                                            bias=bias, LayerNorm_type=LN_type) for i in range(num_blocks_lr[0])])

        self.encoder_L1 = nn.Sequential(*[TransformerBlock(dim=int(dim_l * 2 ** 0), num_heads=heads_en[1], ffn_expansion_factor=ffn_e_f,
                                                            bias=bias, LayerNorm_type=LN_type) for i in range(num_blocks_lr[1])])
        self.down_l1_2 = Downsample(int(dim_l * 2 ** 0))

        self.encoder_L2 = nn.Sequential(*[TransformerBlock(dim=int(dim_l * 2 ** 1), num_heads=heads_en[2], ffn_expansion_factor=ffn_e_f,
                                                            bias=bias, LayerNorm_type=LN_type) for i in range(num_blocks_lr[2])])
        self.down_l2_3 = Downsample(int(dim_l * 2 ** 1))

        self.encoder_L3 = nn.Sequential(*[TransformerBlock(dim=int(dim_l * 2 ** 2), num_heads=heads_en[3], ffn_expansion_factor=ffn_e_f,
                                                            bias=bias, LayerNorm_type=LN_type) for i in range(num_blocks_lr[3])])
        
        ###########################################################
        self.mutual0 = Mutual(hr_channels = dim_h * 2 ** 1, lr_channels = dim_l * 2 ** 0, compressed_channels=32)
        self.mutual1 = Mutual(hr_channels = dim_h * 2 ** 2, lr_channels = dim_l * 2 ** 1, compressed_channels=64)
        self.mutual2 = Mutual(hr_channels = dim_h * 2 ** 3, lr_channels = dim_l * 2 ** 2, compressed_channels=128)

        self.up_cas = Upsample(dim_l * 2 ** 2)
        self.transchannel = nn.Sequential(
            nn.Conv2d(dim_l * 2 ** 1, dim_h  * 2 ** 3, kernel_size=3,stride=1,dilation=1,padding=1),
            nn.BatchNorm2d(int(dim_h * 2 ** 3)),
            nn.ReLU(inplace=True)
            )
        self.DCSAF = DCSAF(dim = dim_h * 2 ** 3, num_heads = 2) #########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def forward(self, lr_feat, hr_feat):
        # C×4H×4W,C×H×W -> 2C×2H×2W,C×H×W
        _hr_0 = self.encoder_H0(hr_feat)
        hr_0 = self.down_h0_1(_hr_0)
        lr_0 = self.encoder_L0(lr_feat)
        hr_0, lr_0 = self.mutual0(hr_feat=hr_0, lr_feat=lr_0) # Freq Mutual
        # 2C×2H×2W,C×H×W -> 4C×H×W,2C×H/2×W/2
        _hr_1 = self.encoder_H1(hr_0)
        hr_1 = self.down_h1_2(_hr_1)
        _lr_1 = self.encoder_L1(lr_0)
        lr_1 = self.down_l1_2(_lr_1)
        before_h, before_l = hr_1, lr_1
        hr_1, lr_1 = self.mutual1(hr_feat=hr_1, lr_feat=lr_1) # Freq Mutual
        after_h, after_l = hr_1, lr_1
        # 4C×H×W,2C×H/2×W/2 -> 8C×H/2×W/2,4C×H/4×W/4
        _hr_2 = self.encoder_H2(hr_1)
        hr_2 = self.down_h2_3(_hr_2)
        _lr_2 = self.encoder_L2(lr_1)
        lr_2 = self.down_l2_3(_lr_2)
        hr_2, lr_2 = self.mutual2(hr_feat=hr_2, lr_feat=lr_2)
        # 8C×H/2×W/2,4C×H/4×W/4
        hr_3 = self.encoder_H3(hr_2)
        lr_3 = self.encoder_L3(lr_2)
        # lr: 4C×H/4×W/4 -> 2C×H/2×W/2 -> 8C×H/2×W/2 ################################################### Cross Sparse Attention
        lr_csa = self.transchannel(self.up_cas(lr_3))
        hr_3 = self.DCSAF(lr_inp = lr_csa, hr_inp = hr_3)

        return hr_0, lr_0, hr_1, hr_2, lr_1, lr_2, hr_3, lr_3



class Decoder(nn.Module):
    def __init__(self,
                 dim_h = c,
                 dim_l = c2,
                 num_blocks_de_hr=[8, 4, 1, 1],
                 num_blocks_de_lr=[2, 2, 2],
                 heads_de=[8, 4, 2],
                 ffn_e_f=2.66,
                 LN_type='WithBias',
                 bias=False,):
        super(Decoder, self).__init__()
        ###########################################################
        self.up_l3_2 = Upsample(int(dim_l * 2 ** 2))
        self.decoder_L3 = nn.Sequential(*[TransformerBlock(dim=int(dim_l * 2 ** 1), num_heads=heads_de[0], ffn_expansion_factor=ffn_e_f,
                                                            bias=bias, LayerNorm_type=LN_type) for i in range(num_blocks_de_lr[0])])

        self.up_l2_1 = Upsample(int(dim_l * 2 ** 1))
        self.decoder_L2 = nn.Sequential(*[TransformerBlock(dim=int(dim_l * 2 ** 0), num_heads=heads_de[1], ffn_expansion_factor=ffn_e_f,
                                                            bias=bias, LayerNorm_type=LN_type) for i in range(num_blocks_de_lr[1])])

        self.decoder_L1 = nn.Sequential(*[TransformerBlock(dim=int(dim_l * 2 ** 0), num_heads=heads_de[2], ffn_expansion_factor=ffn_e_f,
                                                            bias=bias, LayerNorm_type=LN_type) for i in range(num_blocks_de_lr[2])])

        ###########################################################
        self.decoder_H3 = nn.Sequential(*[NAFBlock(c = int(dim_h * 2 ** 3)) for i in range(num_blocks_de_hr[0])])

        self.up_h3_2 = Upsample(int(dim_h * 2 ** 3))
        self.decoder_H2 = nn.Sequential(*[NAFBlock(c = int(dim_h * 2 ** 2)) for i in range(num_blocks_de_hr[1])])

        self.up_h2_1 = Upsample(int(dim_h * 2 ** 2))
        self.decoder_H1 = nn.Sequential(*[NAFBlock(c = int(dim_h * 2 ** 1)) for i in range(num_blocks_de_hr[2])])

        self.up_h1_0 = Upsample(int(dim_h * 2 ** 1))
        self.decoder_H0 = nn.Sequential(*[NAFBlock(c = int(dim_h * 2 ** 0)) for i in range(num_blocks_de_hr[3])])

        ###########################################################
        self.DCTFusion1 = DCTFusion(dim_l = dim_l * 2 ** 1, dim_h = dim_h * 2 ** 3) ############################## DCT Fusion

        self.DCTFusion2 = DCTFusion(dim_l = dim_l * 2 ** 0, dim_h = dim_h * 2 ** 2) ############################## DCT Fusion

        # self.trans_channel1 =nn.Sequential(
        #     nn.Conv2d(dim_l * 2 ** 1, dim_h * 2 ** 3, kernel_size=3,stride=1,dilation=1,padding=1),
        #     nn.BatchNorm2d(dim_h * 2 ** 3),
        #     nn.ReLU(inplace=True)
        #     )


    def forward(self, hr_0, lr_0, hr_1, hr_2, lr_1, lr_2, hr_3, lr_3):
        # 8C×H/2×W/2,4C×H/4×W/4 -> 8C×H/2×W/2,2C×H/2×W/2
        hr_d3 = self.decoder_H3(hr_3)
        _lr_d2 = self.up_l3_2(lr_3 + lr_2) # lr residual
        lr_d2 = self.decoder_L3(_lr_d2)

        hr_d3 = self.DCTFusion1(lr_d2, hr_d3) ############################## DCT Fusion
        hr_d3 =hr_d3 + hr_2 # residual
        # 8C×H/2×W/2,2C×H/2×W/2 -> 4C×H×W,C×H×W
        _hr_d2 = self.up_h3_2(hr_d3)
        hr_d2 = self.decoder_H2(_hr_d2)
        _lr_d1 = self.up_l2_1(lr_d2 + lr_1) # lr residual
        lr_d1 = self.decoder_L2(_lr_d1)

        hr_d2 = self.DCTFusion2(lr_d1, hr_d2) ############################## DCT Fusion
        hr_d2 =hr_d2 + hr_1 # residual
        # 4C×H×W,C×H×W -> 2C×2H×2W,C×H×W
        _hr_d1 = self.up_h2_1(hr_d2)
        hr_d1 = self.decoder_H1(_hr_d1) + hr_0
        _lr_d0 = lr_d1 + lr_0 # lr residual
        lr_d0 = self.decoder_L1(_lr_d0)
        # hr 2C×2H×2W -> C×4H×4W
        _hr_d0 = self.up_h1_0(hr_d1)
        hr_d0 = self.decoder_H0(_hr_d0)

        return lr_d0, hr_d0



class Network(nn.Module):
    def __init__(self, dim_h = c, dim_l = c2):
        super(Network, self).__init__()
        self.hr_proj = nn.Conv2d(3, dim_h, kernel_size=3, padding=1)
        self.lr_proj = nn.Conv2d(3, dim_l, kernel_size=3, padding=1)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.hr_proj_out = nn.Conv2d(dim_h, 3, kernel_size=3, padding=1)
        self.lr_proj_out = nn.Conv2d(dim_l, 3, kernel_size=3, padding=1) 

    def forward(self, x):
        lr_x = F.interpolate(x, size=(x.shape[2] // 4, x.shape[3] // 4), mode='bilinear', align_corners=True)
        # 3×4H×4W,3×H×W -> C×4H×4W,C×H×W
        hr_feat = self.hr_proj(x)
        lr_feat = self.lr_proj(lr_x)

        hr_0, lr_0, hr_1, hr_2, lr_1, lr_2, hr_3, lr_3 = self.encoder(lr_feat, hr_feat)

        lr_d0, hr_d0 = self.decoder(hr_0, lr_0, hr_1, hr_2, lr_1, lr_2, hr_3, lr_3)
        
        hr_d0 = self.hr_proj_out(hr_d0) + x
        lr_d0 = self.lr_proj_out(lr_d0) + lr_x

        return lr_d0, hr_d0



import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

def visualize_conv_fft(weights, layer_name, num_kernels=5):
    """
    对卷积层的权重进行FFT可视化，生成类似提供的图片效果
    
    参数:
        weights: 卷积层权重 (C_out, C_in, kernel_h, kernel_w)
        layer_name: 层名称
        num_kernels: 可视化的卷积核数量
    """
    # 获取输出通道数，限制为num_kernels
    out_channels = min(weights.shape[0], num_kernels)
    
    fig, axes = plt.subplots(2, out_channels, figsize=(5*out_channels, 10))
    if out_channels == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(out_channels):
        # 获取第i个输出通道的所有输入通道的权重
        kernel = weights[i, :, :, :]  # shape: (C_in, kernel_h, kernel_w)
        
        # 将所有输入通道的权重相加，得到一个二维卷积核
        kernel_2d = np.sum(kernel, axis=0)  # shape: (kernel_h, kernel_w)
        
        # 计算2D FFT
        fft_result = fftpack.fft2(kernel_2d)
        fft_shifted = fftpack.fftshift(fft_result)
        
        # 计算幅度谱
        magnitude_spectrum = np.abs(fft_shifted)
        
        # 应用对数变换以增强对比度
        magnitude_log = np.log(magnitude_spectrum + 1e-8)
        
        # 归一化到[0,1]
        magnitude_normalized = (magnitude_log - magnitude_log.min()) / \
                              (magnitude_log.max() - magnitude_log.min())
        
        # 时域卷积核
        axes[0, i].imshow(kernel_2d, cmap='RdBu_r', interpolation='nearest')
        axes[0, i].set_title(f'{layer_name} - Kernel {i} (Spatial)')
        axes[0, i].axis('off')
        
        # 频域幅度谱
        im = axes[1, i].imshow(magnitude_normalized, cmap='viridis', 
                              vmin=0, vmax=1, interpolation='nearest')
        axes[1, i].set_title(f'{layer_name} - Kernel {i} (Frequency)')
        axes[1, i].axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[1, i], shrink=0.8)
        cbar.set_label('Magnitude')
    
    plt.tight_layout()
    plt.show()


def visualize_single_kernel_fft(weight, layer_name):
    """
    可视化单个卷积核的FFT，用于更详细的分析
    """
    # 确保是二维数组
    if weight.ndim == 4:
        # 如果是4D张量，取第一个输出通道和所有输入通道的和
        kernel_2d = np.sum(weight[0, :, :, :], axis=0)
    else:
        kernel_2d = weight
    
    # 计算2D FFT
    fft_result = fftpack.fft2(kernel_2d)
    fft_shifted = fftpack.fftshift(fft_result)
    
    # 计算幅度谱
    magnitude_spectrum = np.abs(fft_shifted)
    
    # 应用对数变换
    magnitude_log = np.log(magnitude_spectrum + 1e-8)
    
    # 归一化
    magnitude_normalized = (magnitude_log - magnitude_log.min()) / \
                          (magnitude_log.max() - magnitude_log.min())
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 时域显示
    im1 = ax1.imshow(kernel_2d, cmap='RdBu_r', interpolation='nearest')
    ax1.set_title(f'{layer_name} - Spatial Domain')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    
    # 频域显示
    im2 = ax2.imshow(magnitude_normalized, cmap='viridis', 
                     vmin=0, vmax=1, interpolation='nearest')
    ax2.set_title(f'{layer_name} - Frequency Domain')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()



def extract_and_visualize_fus_parameters(model_path, model_instance):
    """
    加载模型权重，提取fus模块的content_encoder和content_encoder2参数，并进行FFT可视化
    """
    # 1. 加载模型权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型状态
    resume_state = torch.load(model_path, map_location=device)
    state_dict = resume_state['net']
    model_instance.load_state_dict(state_dict, strict=False)
    model_instance.eval()

    # 2. 提取mutual0模块的content_encoder和content_encoder2参数
    # 直接访问encoder中的mutual0模块
    try:
        fus_module = model_instance.encoder.mutual0
        
        # 提取content_encoder和content_encoder2的权重
        content_encoder_weights = fus_module.content_encoder.weight.detach().cpu().numpy()
        content_encoder2_weights = fus_module.content_encoder2.weight.detach().cpu().numpy()
        
        print(f"Content encoder weights shape: {content_encoder_weights.shape}")
        print(f"Content encoder2 weights shape: {content_encoder2_weights.shape}")
        
        # 3. 对每个卷积层的权重进行FFT可视化
        visualize_single_kernel_fft(content_encoder_weights, "content_encoder")
        visualize_single_kernel_fft(content_encoder2_weights, "content_encoder2")
        
    except AttributeError as e:
        print(f"无法找到mutual0模块: {e}")
        print("可用的模块:")
        for name, module in model_instance.named_modules():
            print(f"  {name}")


# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = Network()
    
    # 替换为实际的模型路径
    model_path = "C:/Users/admin/Desktop/best_990.pth"  # 请替换为实际模型路径
    
    # 可视化卷积核
    extract_and_visualize_fus_parameters(model_path, model)

# ... existing code ...



