import torch
from thop import profile
from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange
import math
try:
    from models._utils import LayerNorm2d
except:
    from _utils import LayerNorm2d




class AdaptiveAttentionFusion(nn.Module):
    def __init__(self, dim_L, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.trans_channel = nn.Conv2d(dim_L, dim, kernel_size=1)
        self.L_norm = LayerNorm2d(dim)
        self.H_norm = LayerNorm2d(dim)
        """fusion L and H"""
        self.L_q_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.L_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.L_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.L_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        self.soft = nn.Softmax(dim = 1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2)) 

    def forward(self, lr_inp, hr_inp):
        lr_inp = self.trans_channel(lr_inp)
        b, c, h, w = lr_inp.shape
        lr = self.L_norm(lr_inp)
        hr = self.H_norm(hr_inp)

        """fusion lr and hr"""
        lr_q = self.L_q_dwconv(self.L_q_conv(lr))
        lr_v = self.L_v_dwconv(self.L_v_conv(lr))
        hr_k = self.H_k_dwconv(self.H_k_conv(hr))
        hr_v = self.H_v_dwconv(self.H_v_conv(hr))

        lr_q = rearrange(lr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        lr_v = rearrange(lr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        hr_k = rearrange(hr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        hr_v = rearrange(hr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        lr_q = torch.nn.functional.normalize(lr_q, dim=-1)
        hr_k = torch.nn.functional.normalize(hr_k, dim=-1)

        attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature

        attn = self.soft(attn)

        out_H = (attn @ hr_v)
        out_L = (attn @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        fusion = out_H*w1 + out_L*w2
        fusion = self.project_out(fusion) + hr_inp

        return fusion


def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,1,2,1,0,0,1,2,3,4,3,2,1,0,0,1,2,3,4,5,6,5,4,3,2,1,0,1,2,3,4]
        all_top_indices_y = [0,1,0,0,1,2,3,2,1,0,0,1,2,3,4,5,4,3,2,1,0,0,1,2,3,4,5,6,6,5,4,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 8, freq_sel_method = 'top16', base_length = 7):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        # self.num_split = len(mapper_x)
        # print(mapper_x, mapper_y)
        mapper_x = [temp_x * (dct_h // base_length) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // base_length) for temp_y in mapper_y]
        # print(mapper_x, mapper_y)
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        # assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # # n, c, h, w = x.shape

        x = x * self.weight
        # print(1 * self.weight)

        result = torch.sum(x, dim=[2,3])
        return result

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):

        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, u_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = \
                        self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, u_y, tile_size_y)
        return dct_filter
        
    def build_filter(self, pos, freq, POS):

        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            # GAP
            return result
        else:
            return result * math.sqrt(2)


class DCTFusion(nn.Module):
    def __init__(self, dim_l, dim_h, h = 128, w = 128, height=2, reduction=8):
        super().__init__()

        self.softmax = nn.Softmax(dim=1)
    
        self.DCTChannel = MultiSpectralAttentionLayer(dim_l +  dim_h, dct_h = h, dct_w = w)

        self.conv1 = nn.Conv2d(dim_l + dim_h, dim_h, kernel_size = 1)
    def forward(self, lr, hr):
        # h, w = hr.shape[2], hr.shape[3]
        feat = torch.cat([lr, hr], dim=1)
        feat = self.DCTChannel(feat)
        out = self.conv1(feat)

        return out

# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                 nn.GELU(),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y


# class RCAB(nn.Module):
#     def __init__(
#         self, n_feat, kernel_size=3, reduction=16,
#         bias=True, bn=False, act=nn.GELU(), res_scale=1):

#         super(RCAB, self).__init__()
#         modules_body = []
#         for i in range(2):
#             modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
#             if i == 0: modules_body.append(act)
#         modules_body.append(CALayer(n_feat, reduction))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale

#     def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
#         return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res

# class CCE(nn.Module):
#     def __init__(self, dim, num_heads=8, bias=False,mode=None):
#         super(CCE, self).__init__()
#         self.num_heads_1 = num_heads
#         self.temperature_1 = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.num_heads_2 = num_heads
#         self.temperature_2 = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv_0_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.qkv_1_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.qkv_2_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

#         self.qkv_0_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.qkv_1_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.qkv_2_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
#         self.qkv1conv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.qkv2conv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
#         self.qkv3conv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
    
#         self.qkv1conv_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.qkv2conv_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
#         self.qkv3conv_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)

#         self.project_out_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.project_out_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

#         self.conv1 = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(1, 2, kernel_size = 3, stride=1, padding=1)
#         self.relu = nn.GELU()
#         self.sigmoid = nn.Sigmoid()
#         self.merge_conv1x1 = nn.Sequential(
#             nn.Conv2d(dim*2, dim, 1, 1), self.relu)
#         self.rcab = RCAB(dim*2)

#     def forward(self, x1, x2):
#         b,c,h,w = x1.shape
#         q1=self.qkv1conv_1(self.qkv_0_1(x1))
#         k1=self.qkv2conv_1(self.qkv_1_1(x1))
#         v1=self.qkv3conv_1(self.qkv_2_1(x1))

#         q2=self.qkv1conv_2(self.qkv_0_2(x2))
#         k2=self.qkv2conv_2(self.qkv_1_2(x2))
#         v2=self.qkv3conv_2(self.qkv_2_2(x2))

#         q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads_1)
#         k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads_1)
#         v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads_1)

#         q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads_2)
#         k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads_2)
#         v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads_2)

#         q1 = torch.nn.functional.normalize(q1, dim=-1)
#         k1 = torch.nn.functional.normalize(k1, dim=-1)

#         q2 = torch.nn.functional.normalize(q2, dim=-1)
#         k2 = torch.nn.functional.normalize(k2, dim=-1)

#         attn1 = (q1 @ k2.transpose(-2, -1)) * self.temperature_1 # q:[4, 8, 8, 7744], k.transpose(-2, -1):[4, 8, 7744, 8]
#         attn1 = attn1.softmax(dim=-1) # [4, 8, 8, 8]
#         out1 = (attn1 @ v2)
#         out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads_1, h=h, w=w)
#         out1 = self.project_out_1(out1)

#         attn2 = (q2 @ k1.transpose(-2, -1)) * self.temperature_2 # q:[4, 8, 8, 7744], k.transpose(-2, -1):[4, 8, 7744, 8]
#         attn2 = attn2.softmax(dim=-1) # [4, 8, 8, 8]
#         out2 = (attn2 @ v1)
#         out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads_2, h=h, w=w)
#         out2 = self.project_out_1(out2)

#         out1 = x1 + out1
#         out2 = x2 + out2
#         out1 = self.relu(out1)
#         out2 = self.relu(out2)

#         rgb_gap = torch.mean(out1, dim=1, keepdim=True)
#         fre_gap = torch.mean(out2, dim=1, keepdim=True)
#         stack_gap = torch.cat([rgb_gap, fre_gap], dim=1)  
#         stack_gap = self.conv1(stack_gap)
#         stack_gap = self.relu(stack_gap)
#         stack_gap = self.conv2(stack_gap)   
#         stack_gap = self.sigmoid(stack_gap)
#         rgb_ = stack_gap[:, 0:1, :, :] * out1 
#         fre_ = stack_gap[:, 1:2, :, :] * out2 
#         merge_feature = torch.cat([rgb_, fre_], dim=1)
#         merge_feature = self.rcab(merge_feature)
#         merge_feature = self.merge_conv1x1(merge_feature)

#         spa_out = (x1 + out1 + merge_feature) / 3
#         spa_out = self.relu(spa_out)
#         return spa_out


class DCSAF(nn.Module):
    def __init__(self, dim, num_heads, height=2, reduction=8, depth = 5, using_diff = True):
        super(DCSAF, self).__init__()
        self.num_heads = num_heads
        self.using_diff = using_diff
        self.L_norm = LayerNorm2d(dim)
        self.H_norm = LayerNorm2d(dim)
        """fusion L and H"""
        self.L_q_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.L_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.L_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.L_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.L_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.L_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_q_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature1 = nn.Parameter(torch.ones(1, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(1, 1, 1))

        self.soft = nn.Softmax(dim = 1)

        # Init λ across heads
        self.head_size = dim // num_heads
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)

        # self.conv1 = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(1, 2, kernel_size = 3, stride=1, padding=1)
        # self.relu = nn.GELU()
        # self.sigmoid = nn.Sigmoid()

        self.conv_out = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1, 1), nn.GELU())


    def forward(self, lr_inp, hr_inp):
        b, c, h, w = hr_inp.shape
        lr = self.L_norm(lr_inp)
        hr = self.H_norm(hr_inp)

        """fusion lr and hr"""
        lr_q = self.L_q_dwconv(self.L_q_conv(lr))
        lr_k = self.L_k_dwconv(self.L_k_conv(lr))
        lr_v = self.L_v_dwconv(self.L_v_conv(lr))
        hr_q = self.H_q_dwconv(self.H_q_conv(hr))
        hr_k = self.H_k_dwconv(self.H_k_conv(hr))
        hr_v = self.H_v_dwconv(self.H_v_conv(hr))

        lr_q = rearrange(lr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        lr_k = rearrange(lr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        lr_v = rearrange(lr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        hr_q = rearrange(hr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        hr_k = rearrange(hr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        hr_v = rearrange(hr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        lr_q = torch.nn.functional.normalize(lr_q, dim=-1)
        lr_k = torch.nn.functional.normalize(lr_k, dim=-1)
        hr_k = torch.nn.functional.normalize(hr_k, dim=-1)
        hr_q = torch.nn.functional.normalize(hr_q, dim=-1)

        attnh = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature1 # b head (c/head) (c/head)
        attnl = (hr_q @ lr_k.transpose(-2, -1)) * self.temperature2
        attn = attnh

        attnh = self.soft(attnh)
        attnl = self.soft(attnl)

        # Compute λ for each head separately
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_H = lambda_1 - lambda_2 + self.lambda_init
        lambda_L = lambda_2 - lambda_1 + self.lambda_init

        if self.using_diff:
            att_H = attnh - lambda_H * attnl
            att_L = attnl - lambda_L * attnh

            out_H = (att_H @ hr_v) * (1 - self.lambda_init)
            out_L = (att_L @ lr_v) * (1 - self.lambda_init)
        else:
            attn1 = self.soft(attn)
            attn2 = self.soft(attn. transpose(-2, -1))
            out_H = attn1 @ hr_v
            out_L = attn2 @ lr_v

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = torch.cat([out_H, out_L], dim=1)
        fusion = self.conv_out(out) + hr_inp

        return fusion