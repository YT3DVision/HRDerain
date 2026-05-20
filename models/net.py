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



if __name__ == '__main__':
    dim =3
    H = 256 # 3840
    W = 256 # 2160
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cuda:0'
    print("start")


    # lr = torch.randn(1, 32, H // 4, W // 4).to(device)
    # hr = torch.randn(1, 32, H, W).to(device)
    # net2 = Encoder().to(device)
    # flops, params = profile(net2, inputs=(lr,hr))
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))
    # Number of parameters:6.0372 M

    net = Network().to(device)
    net.eval()
    # input_tensor = torch.randn(1, 3, 1024, 1024)
    # number_iter = 20
    x = torch.randn(1, dim, H, W).to(device)
    flops, params = profile(net, inputs=(x,))
    print(' Number of parameters:%.4f M' % (params / 1e6))
    print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))
    #  Number of parameters:12.4950 M    Number of FLOPs:18.5498 GFLOPs

    # input_tensor = input_tensor.to(device)
    # for _ in range(10):
    #     with torch.no_grad():
    #         net(input_tensor)

    # start_time = time.time()
    # for _ in range(number_iter):
    #     with torch.no_grad():
    #         net(input_tensor)
    # end_time = time.time()
    # average_time = (end_time - start_time) / number_iter
    # print(average_time)






    print("end")

