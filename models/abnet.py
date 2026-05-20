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
    from models._utils import LayerNorm2d
except:
    from NAFBlock import NAFBlock
    from restormer_arch import TransformerBlock, Upsample, Downsample
    from simFreq import fus as Mutual
    from AttentionFusion import DCSAF, DCTFusion
    from _utils import LayerNorm2d

from einops import rearrange
from thop import profile






class CA(nn.Module):
    def __init__(self, dim, num_heads):
        super(CA, self).__init__()
        self.num_heads = num_heads
        self.L_norm = LayerNorm2d(dim)
        self.H_norm = LayerNorm2d(dim)
        """fusion L and H"""
        self.L_q_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.L_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature1 = nn.Parameter(torch.ones(1, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(1, 1, 1))

        self.soft = nn.Softmax(dim = 1)

    def forward(self, lr_inp, hr_inp):
        b, c, h, w = hr_inp.shape
        lr = self.L_norm(lr_inp)
        hr = self.H_norm(hr_inp)

        """fusion lr and hr"""
        lr_q = self.L_q_dwconv(self.L_q_conv(lr))
        hr_k = self.H_k_dwconv(self.H_k_conv(hr))
        hr_v = self.H_v_dwconv(self.H_v_conv(hr))

        lr_q = rearrange(lr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        hr_k = rearrange(hr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        hr_v = rearrange(hr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        lr_q = torch.nn.functional.normalize(lr_q, dim=-1)
        hr_k = torch.nn.functional.normalize(hr_k, dim=-1)

        attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature1

        attn = self.soft(attn)
        out_H = attn @ hr_v

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        final = out_H + hr_inp

        return final



class Fusion_sum(nn.Module):
    def __init__(self, dim_l, dim_h):
        super(Fusion_sum, self).__init__()
        self.conv1 = nn.Conv2d(dim_l, dim_h, kernel_size=1)
        # self.conv2 = nn.Conv2d(dim_h, dim_h, kernel_size=3, padding=1)
    
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        # x2 = self.conv2(x2)

        x = x1 + x2

        return x


class Fusion_cat(nn.Module):
    def __init__(self, dim_l, dim_h):
        super(Fusion_cat, self).__init__()
        # self.conv1 = nn.Conv2d(dim_l, dim_l, kernel_size=1)
        # self.conv2 = nn.Conv2d(dim_h, dim_h, kernel_size=1)
        self.conv3 = nn.Conv2d(dim_l + dim_h, dim_h, kernel_size=1)
    
    def forward(self, x1, x2):
        # x1 = self.conv1(x1)
        # x2 = self.conv2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.conv3(x)

        return x


class no_Mutual(nn.Module):
    def __init__(self, hr_channels, lr_channels,compressed_channels):
        super(no_Mutual, self).__init__()


    def forward(self, hr_feat, lr_feat):
        return hr_feat, lr_feat



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
        hr_1, lr_1 = self.mutual1(hr_feat=hr_1, lr_feat=lr_1) # Freq Mutual
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



if __name__ == '__main__':
    dim =3
    H = 256 # 3840
    W = 256 # 2160
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("start")


    net = Network().to(device)
    net.eval()
    x = torch.randn(1, dim, H, W).to(device)
    flops, params = profile(net, inputs=(x,))
    print(' Number of parameters:%.4f M' % (params / 1e6))
    print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))


    print("end")


# ????????????????????????????????????????????????????????????????????
# 10  Number of parameters:12.5449 M    Number of FLOPs:18.6840 GFLOPs
# 11  Number of parameters:13.2822 M   Number of FLOPs:19.8920 GFLOPs
# 12  Number of parameters:12.4950 M    Number of FLOPs:18.5498 GFLOPs







# 1   Number of parameters:11.9111 M    Number of FLOPs:17.6087 GFLOPs
# 2   Number of parameters:11.9930 M    Number of FLOPs:17.7429 GFLOPs
# 3   Number of parameters:12.0250 M    Number of FLOPs:17.7429 GFLOPs

# 4   Number of parameters:12.0447 M    Number of FLOPs:18.0729 GFLOPs
# 5   Number of parameters:12.1266 M    Number of FLOPs:18.2071 GFLOPs
# 6   Number of parameters:12.1586 M    Number of FLOPs:18.2071 GFLOPs



# 7   Number of parameters:12.2475 M    Number of FLOPs:17.9514 GFLOPs 
# 8   Number of parameters:12.3294 M    Number of FLOPs:18.0856 GFLOPs
# 9   Number of parameters:12.3614 M    Number of FLOPs:18.0856 GFLOPs

# 10  Number of parameters:12.3811 M    Number of FLOPs:18.4156 GFLOPs
# 11  Number of parameters:12.4630 M    Number of FLOPs:18.5498 GFLOPs
# 12  Number of parameters:12.4950 M    Number of FLOPs:18.5498 GFLOPs

