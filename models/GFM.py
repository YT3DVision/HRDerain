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

class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()

		self.height = height
		d = max(int(dim/reduction), 4)

		self.mlp = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(True),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape
		in_feats = torch.cat(in_feats, dim=1)
	# def forward(self, in_feats0 , in_feats1):
	# 	B, C, H, W = in_feats0.shape
	# 	in_feats = torch.cat((in_feats0, in_feats1), dim=1)

		in_feats = in_feats.view(B, self.height, C, H, W)

		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(feats_sum)
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out


"""Grafting Fusion Module"""
class GFM(nn.Module):
    def __init__(self,dim,num_heads):
        super(GFM, self).__init__()
        self.num_heads = num_heads
        self.T_norm = LayerNorm2d(dim)
        self.C_norm = LayerNorm2d(dim)
        """fusion T_E and C_E"""
        self.T_q_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.T_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.C_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.C_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.C_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.C_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.SK_fusion = SKFusion(dim = dim)

    def forward(self,T_E_input,C_E_input,T_D_input):
        b, c, h, w = T_E_input.shape
        T_E = self.T_norm(T_E_input)
        C_E = self.C_norm(C_E_input)

        """fusion T_E and C_E"""
        T_E_q = self.T_q_dwconv(self.T_q_conv(T_E))
        T_C_k = self.C_k_dwconv(self.C_k_conv(C_E))
        T_C_v = self.C_v_dwconv(self.C_v_conv(C_E))

        T_E_q = rearrange(T_E_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        T_C_k = rearrange(T_C_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        T_C_v = rearrange(T_C_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        T_E_q = torch.nn.functional.normalize(T_E_q, dim=-1)
        T_C_k = torch.nn.functional.normalize(T_C_k, dim=-1)

        _,_,C,_ = T_E_q.shape
        """top_k mask"""
        mask1 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)

        attn = (T_E_q @ T_C_k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ T_C_v)
        out2 = (attn2 @ T_C_v)
        out3 = (attn3 @ T_C_v)
        out4 = (attn4 @ T_C_v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        fusion1 = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion1 = self.project_out(fusion1) + C_E_input

        """fusion E and decoder"""
        fusion2 = self.SK_fusion([fusion1,T_D_input])

        return fusion2

"""Cross Sparse Attention"""
class CSA(nn.Module):
    def __init__(self,dim,num_heads):
        super(CSA, self).__init__()
        self.num_heads = num_heads
        self.T_norm = LayerNorm2d(dim)
        self.C_norm = LayerNorm2d(dim)
        """fusion T_E and C_E"""
        self.T_q_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.T_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.C_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.C_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.C_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.C_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)


    def forward(self,T_E_input,C_E_input):
        b, c, h, w = T_E_input.shape
        T_E = self.T_norm(T_E_input)
        C_E = self.C_norm(C_E_input)

        """fusion T_E and C_E"""
        T_E_q = self.T_q_dwconv(self.T_q_conv(T_E))
        T_C_k = self.C_k_dwconv(self.C_k_conv(C_E))
        T_C_v = self.C_v_dwconv(self.C_v_conv(C_E))

        T_E_q = rearrange(T_E_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        T_C_k = rearrange(T_C_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        T_C_v = rearrange(T_C_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        T_E_q = torch.nn.functional.normalize(T_E_q, dim=-1)
        T_C_k = torch.nn.functional.normalize(T_C_k, dim=-1)

        _,_,C,_ = T_E_q.shape
        """top_k mask"""
        mask1 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)

        attn = (T_E_q @ T_C_k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ T_C_v)
        out2 = (attn2 @ T_C_v)
        out3 = (attn3 @ T_C_v)
        out4 = (attn4 @ T_C_v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        fusion = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = self.project_out(fusion) + C_E_input

        return fusion


"""Dual Cross Sparse Attention"""
class DCSA(nn.Module):
    def __init__(self,dim,num_heads):
        super(DCSA, self).__init__()
        self.num_heads = num_heads
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

        self.project_out_H = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out_L = nn.Conv2d(dim, dim, kernel_size=1)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)


    def forward(self,lr,hr):
        b, c, h, w = lr.shape
        lr = self.L_norm(lr)
        hr = self.H_norm(hr)

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

        _,_,C,_ = lr_q.shape
        """top_k mask"""
        # mask1 = torch.zeros(b, self.num_heads, C, C, device=lr.device, requires_grad=False)
        # mask2 = torch.zeros(b, self.num_heads, C, C, device=lr.device, requires_grad=False)
        # mask3 = torch.zeros(b, self.num_heads, C, C, device=lr.device, requires_grad=False)
        # mask4 = torch.zeros(b, self.num_heads, C, C, device=lr.device, requires_grad=False)

        # attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature

        # index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        # mask1.scatter_(-1, index, 1.)
        # attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        # index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        # mask2.scatter_(-1, index, 1.)
        # attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        # index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        # mask3.scatter_(-1, index, 1.)
        # attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        # index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        # mask4.scatter_(-1, index, 1.)
        # attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        # attn1 = attn1.softmax(dim=-1)
        # attn2 = attn2.softmax(dim=-1)
        # attn3 = attn3.softmax(dim=-1)
        # attn4 = attn4.softmax(dim=-1)

        # out1 = (attn1 @ hr_v)
        # out2 = (attn2 @ hr_v)
        # out3 = (attn3 @ hr_v)
        # out4 = (attn4 @ hr_v)

        # out5 = (attn1 @ lr_v)
        # out6 = (attn2 @ lr_v)
        # out7 = (attn3 @ lr_v)
        # out8 = (attn4 @ lr_v)

        # out_H = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
        # out_L = out5 * self.attn1 + out6 * self.attn2 + out7 * self.attn3 + out8 * self.attn4

        attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # print("lr_q",lr_q.shape)
        # print("hr_k",hr_k.shape)
        print("attn",attn.shape)
        out_H = (attn @ hr_v)
        out_L = (attn @ lr_v)

        fusion_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        fusion_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion_H = self.project_out_H(fusion_H) + hr
        fusion_L = self.project_out_L(fusion_L) + lr

        return fusion_L, fusion_H


"""Dual Cross Select Attention Fusion"""
# class DCSAF(nn.Module):
#     def __init__(self, dim, num_heads, height=2, reduction=8):
#         super(DCSAF, self).__init__()
#         self.num_heads = num_heads
#         self.L_norm = LayerNorm2d(dim)
#         self.H_norm = LayerNorm2d(dim)
#         """fusion L and H"""
#         self.L_q_conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.L_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

#         self.L_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.L_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

#         self.H_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.H_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

#         self.H_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.H_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

#         self.temperature = nn.Parameter(torch.ones(1, 1, 1))

#         self.project_out_H = nn.Conv2d(dim, dim, kernel_size=1)
#         self.project_out_L = nn.Conv2d(dim, dim, kernel_size=1)

#         self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
#         self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
#         self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
#         self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

#         temp = int((dim * dim )/(self.num_heads * self.num_heads))
#         d = max(int(temp/reduction), 4)
#         self.squeeze1 = nn.Conv2d(temp, d, 1, bias=False)
#         self.relu = nn.ReLU(True)
#         self.squeeze2 = nn.Conv2d(d, temp*height, 1, bias=False)
#         self.soft = nn.Softmax(dim=1)


#     def forward(self,lr,hr):
#         b, c, h, w = lr.shape
#         lr = self.L_norm(lr)
#         hr = self.H_norm(hr)

#         """fusion lr and hr"""
#         lr_q = self.L_q_dwconv(self.L_q_conv(lr))
#         lr_v = self.L_v_dwconv(self.L_v_conv(lr))
#         hr_k = self.H_k_dwconv(self.H_k_conv(hr))
#         hr_v = self.H_v_dwconv(self.H_v_conv(hr))

#         lr_q = rearrange(lr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         lr_v = rearrange(lr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         hr_k = rearrange(hr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         hr_v = rearrange(hr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         lr_q = torch.nn.functional.normalize(lr_q, dim=-1)
#         hr_k = torch.nn.functional.normalize(hr_k, dim=-1)

#         _,_,C,_ = lr_q.shape
#         """top_k mask"""
#         # mask1 = torch.zeros(b, self.num_heads, C, C, device=lr.device, requires_grad=False)
#         # mask2 = torch.zeros(b, self.num_heads, C, C, device=lr.device, requires_grad=False)
#         # mask3 = torch.zeros(b, self.num_heads, C, C, device=lr.device, requires_grad=False)
#         # mask4 = torch.zeros(b, self.num_heads, C, C, device=lr.device, requires_grad=False)

#         # attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature

#         # index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
#         # mask1.scatter_(-1, index, 1.)
#         # attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

#         # index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
#         # mask2.scatter_(-1, index, 1.)
#         # attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

#         # index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
#         # mask3.scatter_(-1, index, 1.)
#         # attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

#         # index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
#         # mask4.scatter_(-1, index, 1.)
#         # attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

#         # attn1 = attn1.softmax(dim=-1)
#         # attn2 = attn2.softmax(dim=-1)
#         # attn3 = attn3.softmax(dim=-1)
#         # attn4 = attn4.softmax(dim=-1)

#         # out1 = (attn1 @ hr_v)
#         # out2 = (attn2 @ hr_v)
#         # out3 = (attn3 @ hr_v)
#         # out4 = (attn4 @ hr_v)

#         # out5 = (attn1 @ lr_v)
#         # out6 = (attn2 @ lr_v)
#         # out7 = (attn3 @ lr_v)
#         # out8 = (attn4 @ lr_v)

#         # out_H = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
#         # out_L = out5 * self.attn1 + out6 * self.attn2 + out7 * self.attn3 + out8 * self.attn4

#         attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature
#         # attn = attn.softmax(dim=-1) # b head (c/head) (c/head)

#         _,_,ccc,_ = attn.shape
#         attn = rearrange(attn, 'b head ch cH -> b head (ch  cH) 1', head = self.num_heads, ch = ccc)
#         attn = attn.permute(0, 2, 1, 3) # b (c*c/head*head) head 1

#         attn = self.squeeze1(attn) #  b (c*c/head*head*reduction) head 1
#         attn = self.relu(attn)
#         attn = self.squeeze2(attn) # b (2*c*c/head*head) head 1
#         attn = self.soft(attn)

#         attn = attn.permute(0, 2, 1, 3) # b head (2*c*c/head*head) 1
#         attnL, attnH = torch.chunk(attn, 2, dim=2)
#         attnL = rearrange(attnL, 'b head (ch cH) 1 -> b head ch cH', head=self.num_heads, ch = ccc)
#         attnH = rearrange(attnH, 'b head (ch cH) 1 -> b head ch cH', head=self.num_heads, ch = ccc)

#         out_H = (attnH @ hr_v)
#         out_L = (attnL @ lr_v)

#         out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         fusion = out_H + out_L
#         fusion_L = self.project_out_L(fusion) + lr
#         fusion_H = self.project_out_H(fusion) + hr

#         return fusion_L, fusion_H



####################################################################
class DCSAF(nn.Module):
    def __init__(self, dim, num_heads, height=2, reduction=8):
        super(DCSAF, self).__init__()
        self.num_heads = num_heads
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

        self.project_out_H = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out_L = nn.Conv2d(dim, dim, kernel_size=1)

        temp = int(dim/self.num_heads )
        d = max(int(temp/reduction), 4)
        self.squeeze1 = nn.Conv2d(temp, d, 1, bias=False)
        self.relu = nn.ReLU(True)
        self.squeeze2 = nn.Conv2d(d, temp*height, 1, bias=False)
        self.soft = nn.Softmax(dim = 1)


    def forward(self, lr_inp, hr_inp):
        b, c, h, w = lr.shape
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
        # attn = attn.softmax(dim=-1) # b head (c/head) (c/head)

        attn = attn.permute(0, 3, 2, 1) # b (c/head) (c/head) head

        attn = self.squeeze1(attn) # b (c/head*reduction) (c/head) head
        attn = self.relu(attn)
        attn = self.squeeze2(attn) # b (2c/head) (c/head) head
        attn = self.soft(attn) # softmax:(2c/head)

        attn = attn.permute(0, 3, 2, 1) # b head (c/head) (2c/head)
        attnL, attnH = torch.chunk(attn, 2, dim=3)

        out_H = (attnH @ hr_v)
        out_L = (attnL @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = out_H + out_L
        fusion_L = self.project_out_L(fusion) + lr_inp
        fusion_H = self.project_out_H(fusion) + hr_inp

        return fusion_L, fusion_H







# for net8
class DCSAF2(nn.Module):
    def __init__(self, dim_L, dim, num_heads, height=2, reduction=8):
        super(DCSAF2, self).__init__()
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

        temp = int(dim/self.num_heads )
        d = max(int(temp/reduction), 4)
        self.squeeze1 = nn.Conv2d(temp, d, 1, bias=False)
        self.relu = nn.ReLU(True)
        self.squeeze2 = nn.Conv2d(d, temp*height, 1, bias=False)
        self.soft = nn.Softmax(dim = 1)


    def forward(self, lr_inp, hr_inp):
        lr = self.trans_channel(lr_inp)
        b, c, h, w = lr.shape
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

        attn = attn.permute(0, 3, 2, 1) # b (c/head) (c/head) head

        attn = self.squeeze1(attn) # b (c/head*reduction) (c/head) head
        attn = self.relu(attn)
        attn = self.squeeze2(attn) # b (2c/head) (c/head) head
        attn = self.soft(attn) # softmax:(2c/head)

        attn = attn.permute(0, 3, 2, 1) # b head (c/head) (2c/head)
        attnL, attnH = torch.chunk(attn, 2, dim=3)

        out_H = (attnH @ hr_v)
        out_L = (attnL @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = out_H + out_L
        fusion = self.project_out(fusion) + hr_inp

        return fusion


# for net8

class DCSAF3(nn.Module):
    def __init__(self, dim_L, dim, num_heads, height=2, reduction=8):
        super(DCSAF3, self).__init__()
        self.num_heads = num_heads
        self.trans_channel = nn.Conv2d(dim, dim_L, kernel_size=1)
        self.L_norm = LayerNorm2d(dim_L)
        self.H_norm = LayerNorm2d(dim_L)
        """fusion L and H"""
        self.L_q_conv = nn.Conv2d(dim_L, dim_L, kernel_size=1)
        self.L_q_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim_L)

        self.L_v_conv = nn.Conv2d(dim_L, dim_L, kernel_size=1)
        self.L_v_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim_L)

        self.H_k_conv = nn.Conv2d(dim_L, dim_L, kernel_size=1)
        self.H_k_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim_L)

        self.H_v_conv = nn.Conv2d(dim_L, dim_L, kernel_size=1)
        self.H_v_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim_L)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.project_out = nn.Conv2d(dim_L, dim_L, kernel_size=1)

        temp = int(dim_L/self.num_heads )
        d = max(int(temp/reduction), 4)
        self.squeeze1 = nn.Conv2d(temp, d, 1, bias=False)
        self.relu = nn.ReLU(True)
        self.squeeze2 = nn.Conv2d(d, temp*height, 1, bias=False)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, lr_inp, hr_inp):
        hr = self.trans_channel(hr_inp)
        b, c, h, w = hr.shape
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
        # attn = attn.softmax(dim=-1) # b head (c/head) (c/head)

        attn = attn.permute(0, 3, 2, 1) # b (c/head) (c/head) head

        attn = self.squeeze1(attn) # b (c/head*reduction) (c/head) head
        attn = self.relu(attn)
        attn = self.squeeze2(attn) # b (2c/head) (c/head) head
        attn = self.soft(attn) # softmax:(2c/head)

        attn = attn.permute(0, 3, 2, 1) # b head (c/head) (2c/head)
        attnL, attnH = torch.chunk(attn, 2, dim=3)

        out_H = (attnH @ hr_v)
        out_L = (attnL @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = out_H + out_L
        fusion = self.project_out(fusion) + lr_inp

        return fusion





############### Efficient_DCSAF #################


class Efficient_DCSAF(nn.Module):
    def __init__(self, dim, num_heads, height=2,):
        super(Efficient_DCSAF, self).__init__()
        self.num_heads = num_heads
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

        self.project_out_H = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out_L = nn.Conv2d(dim, dim, kernel_size=1)

        # Efficient Channel Attention
        c=int(dim/self.num_heads )
        b=1
        gamma=2
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.efficient_CA_conv = nn.Conv2d(self.num_heads, int(height*self.num_heads), kernel_size=k, padding=int(k/2), bias=False)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, lr_inp, hr_inp):
        b, c, h, w = lr.shape
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

        attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature # b head (c/head) (c/head)

        attn = self.efficient_CA_conv(attn) # b 2*head (c/head) (c/head)
        attn = self.soft(attn)

        attnL, attnH = torch.chunk(attn, 2, dim=1)

        out_H = (attnH @ hr_v)
        out_L = (attnL @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = out_H + out_L
        fusion_L = self.project_out_L(fusion) + lr_inp
        fusion_H = self.project_out_H(fusion) + hr_inp

        return fusion_L, fusion_H


class Efficient_DCSAF2(nn.Module):
    def __init__(self, dim_L, dim , num_heads, height=2,):
        super(Efficient_DCSAF2, self).__init__()
        self.num_heads = num_heads
        self.L_norm = LayerNorm2d(dim_L)
        self.H_norm = LayerNorm2d(dim)
        """fusion L and H"""
        self.L_q_conv = nn.Conv2d(dim_L, dim, kernel_size=1)
        self.L_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.L_v_conv = nn.Conv2d(dim_L, dim, kernel_size=1)
        self.L_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.project_out_L = nn.Conv2d(dim, dim_L, kernel_size=1)
        self.down_L = nn.Conv2d(dim_L, int(dim_L/2), kernel_size=1)
        self.project_out_H = nn.Conv2d(dim, dim, kernel_size=1)


        # Efficient Channel Attention
        c=int(dim/self.num_heads )
        b=1
        gamma=2
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.efficient_CA_conv = nn.Conv2d(self.num_heads, int(height*self.num_heads), kernel_size=k, padding=int(k/2), bias=False)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, lr_inp, hr_inp):
        b, c, h, w = lr.shape
        lr = self.L_norm(lr_inp)
        hr = self.H_norm(hr_inp)

        """fusion lr and hr"""
        lr_q = self.L_q_dwconv(self.L_q_conv(lr))
        lr_v = self.L_v_dwconv(self.L_v_conv(lr))
        hr_k = self.H_k_dwconv(self.H_k_conv(hr))
        hr_v = self.H_v_dwconv(self.H_v_conv(hr))

        lr_q = rearrange(lr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        lr_v = rearrange(lr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_k = rearrange(hr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_v = rearrange(hr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)

        lr_q = torch.nn.functional.normalize(lr_q, dim=-1)
        hr_k = torch.nn.functional.normalize(hr_k, dim=-1)

        attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature # b head (c/head) (c/head)

        attn = self.efficient_CA_conv(attn) # b 2*head (c/head) (c/head)
        attn = self.soft(attn)

        attnL, attnH = torch.chunk(attn, 2, dim=1)

        out_H = (attnH @ hr_v)
        out_L = (attnL @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = out_H + out_L
        fusion_L = self.project_out_L(fusion) + lr_inp
        fusion_L = self.down_L(fusion_L)
        fusion_H = self.project_out_H(fusion) + hr_inp

        return fusion_L, fusion_H





# net7
class Efficient_DCSAF3(nn.Module):
    def __init__(self, dim_L, dim , num_heads, height=2,):
        super(Efficient_DCSAF3, self).__init__()
        self.num_heads = num_heads
        self.L_norm = LayerNorm2d(dim_L)
        self.H_norm = LayerNorm2d(dim)
        """fusion L and H"""
        self.L_q_conv = nn.Conv2d(dim_L, dim, kernel_size=1)
        self.L_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.L_v_conv = nn.Conv2d(dim_L, dim, kernel_size=1)
        self.L_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.project_out_L = nn.Conv2d(dim, int(dim/4), kernel_size=1)
        self.project_out_H = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out_fusion = nn.Conv2d(dim, int(dim/2), kernel_size=1)


        # Efficient Channel Attention
        c=int(dim/self.num_heads )
        b=1
        gamma=2
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.efficient_CA_conv = nn.Conv2d(self.num_heads, int(height*self.num_heads), kernel_size=k, padding=int(k/2), bias=False)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, lr_inp, hr_inp):
        b, c, h, w = lr.shape
        lr = self.L_norm(lr_inp)
        hr = self.H_norm(hr_inp)

        """fusion lr and hr"""
        lr_q = self.L_q_dwconv(self.L_q_conv(lr))
        lr_v = self.L_v_dwconv(self.L_v_conv(lr))
        hr_k = self.H_k_dwconv(self.H_k_conv(hr))
        hr_v = self.H_v_dwconv(self.H_v_conv(hr))

        lr_q = rearrange(lr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        lr_v = rearrange(lr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_k = rearrange(hr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_v = rearrange(hr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)

        lr_q = torch.nn.functional.normalize(lr_q, dim=-1)
        hr_k = torch.nn.functional.normalize(hr_k, dim=-1)

        attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature # b head (c/head) (c/head)

        attn = self.efficient_CA_conv(attn) # b 2*head (c/head) (c/head)
        attn = self.soft(attn)

        attnL, attnH = torch.chunk(attn, 2, dim=1)

        out_H = (attnH @ hr_v)
        out_L = (attnL @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = out_H + out_L
        # fusion_L = self.project_out_L(fusion) + lr
        # fusion_H = self.project_out_H(fusion) + hr
        fusion_L = self.project_out_L(fusion)
        fusion_H = self.project_out_H(fusion)
        fusion = self.project_out_fusion(fusion)

        return fusion_L, fusion_H, fusion





# net8
class ____________Efficient_DCSAF4(nn.Module):
    def __init__(self, dim_L, dim , num_heads, height=2):
        super(____________Efficient_DCSAF4, self).__init__()
        self.num_heads = num_heads
        self.L_norm = LayerNorm2d(dim_L)
        self.H_norm = LayerNorm2d(dim)
        """fusion L and H"""
        self.L_q_conv = nn.Conv2d(dim_L, dim, kernel_size=1)
        self.L_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.L_v_conv = nn.Conv2d(dim_L, dim, kernel_size=1)
        self.L_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.H_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)


        # Efficient Channel Attention
        c=int(dim/self.num_heads )
        b=1
        gamma=2
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.efficient_CA_conv = nn.Conv2d(self.num_heads, int(height*self.num_heads), kernel_size=k, padding=int(k/2), bias=False)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, lr_inp, hr_inp):
        b, c, h, w = lr_inp.shape
        lr = self.L_norm(lr_inp)
        hr = self.H_norm(hr_inp)

        """fusion lr and hr"""
        lr_q = self.L_q_dwconv(self.L_q_conv(lr))
        lr_v = self.L_v_dwconv(self.L_v_conv(lr))
        hr_k = self.H_k_dwconv(self.H_k_conv(hr))
        hr_v = self.H_v_dwconv(self.H_v_conv(hr))

        lr_q = rearrange(lr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        lr_v = rearrange(lr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_k = rearrange(hr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_v = rearrange(hr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)

        lr_q = torch.nn.functional.normalize(lr_q, dim=-1)
        hr_k = torch.nn.functional.normalize(hr_k, dim=-1)

        attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature # b head (c/head) (c/head)

        attn = self.efficient_CA_conv(attn) # b 2*head (c/head) (c/head)
        attn = self.soft(attn)

        attnL, attnH = torch.chunk(attn, 2, dim=1)

        out_H = (attnH @ hr_v)
        out_L = (attnL @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = out_H + out_L
        fusion = self.project_out(fusion) + hr_inp

        return fusion

# net8
class ____________Efficient_DCSAF5(nn.Module):
    def __init__(self, dim_L, dim , num_heads, height=2):
        super(____________Efficient_DCSAF4, self).__init__()
        self.num_heads = num_heads
        self.L_norm = LayerNorm2d(dim_L)
        self.H_norm = LayerNorm2d(dim)
        """fusion L and H"""
        self.L_q_conv = nn.Conv2d(dim_L, dim_L, kernel_size=1)
        self.L_q_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim)

        self.L_v_conv = nn.Conv2d(dim_L, dim_L, kernel_size=1)
        self.L_v_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_k_conv = nn.Conv2d(dim, dim_L, kernel_size=1)
        self.H_k_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim)

        self.H_v_conv = nn.Conv2d(dim, dim_L, kernel_size=1)
        self.H_v_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.project_out = nn.Conv2d(dim_L, dim_L, kernel_size=1)


        # Efficient Channel Attention
        c = int(dim_L/self.num_heads)
        b = 1
        gamma = 2
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.efficient_CA_conv = nn.Conv2d(self.num_heads, int(height*self.num_heads), kernel_size=k, padding=int(k/2), bias=False)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, lr_inp, hr_inp):
        b, c, h, w = lr_inp.shape
        lr = self.L_norm(lr_inp)
        hr = self.H_norm(hr_inp)

        """fusion lr and hr"""
        lr_q = self.L_q_dwconv(self.L_q_conv(lr))
        lr_v = self.L_v_dwconv(self.L_v_conv(lr))
        hr_k = self.H_k_dwconv(self.H_k_conv(hr))
        hr_v = self.H_v_dwconv(self.H_v_conv(hr))

        lr_q = rearrange(lr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        lr_v = rearrange(lr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_k = rearrange(hr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_v = rearrange(hr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)

        lr_q = torch.nn.functional.normalize(lr_q, dim=-1)
        hr_k = torch.nn.functional.normalize(hr_k, dim=-1)

        attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature # b head (c/head) (c/head)

        attn = self.efficient_CA_conv(attn) # b 2*head (c/head) (c/head)
        attn = self.soft(attn)

        attnL, attnH = torch.chunk(attn, 2, dim=1)

        out_H = (attnH @ hr_v)
        out_L = (attnL @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = out_H + out_L
        fusion = self.project_out(fusion) + lr_inp

        return fusion



###      ###

# net8 new, two efficient_CA_conv
class Efficient_DCSAF4(nn.Module):
    def __init__(self, dim_L, dim , num_heads):
        super(Efficient_DCSAF4, self).__init__()
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


        # Efficient Channel Attention
        ch = int(dim/self.num_heads )
        cl = int(dim_L/self.num_heads)
        b=1
        gamma=2
        th = int(abs((math.log(ch, 2) + b) / gamma))
        tl = int(abs((math.log(cl, 2) + b) / gamma))
        kh = th if th % 2 else th + 1
        kl = tl if tl % 2 else tl + 1
        self.efficient_CA_conv_H = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=kh, padding=int(kh/2), bias=False)
        self.efficient_CA_conv_L = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=kl, padding=int(kl/2), bias=False)
        self.soft = nn.Softmax(dim = -1)

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

        lr_q = rearrange(lr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        lr_v = rearrange(lr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_k = rearrange(hr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_v = rearrange(hr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)

        lr_q = torch.nn.functional.normalize(lr_q, dim=-1)
        hr_k = torch.nn.functional.normalize(hr_k, dim=-1)

        attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature # b head (c/head) (c/head)

        attnh = self.efficient_CA_conv_H(attn) # b head (c/head) (c/head)
        attnl = self.efficient_CA_conv_L(attn) # b head (c/head) (c/head)
        temp = torch.cat((attnh, attnl), dim = 3)
        temp = self.soft(temp)
        attnL, attnH = torch.chunk(temp, 2, dim=3)

        out_H = (attnH @ hr_v)
        out_L = (attnL @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = out_H + out_L
        fusion = self.project_out(fusion) + hr_inp

        return fusion



# c = 32 * 8
# b=1
# gamma=2
# t = int(abs((math.log(c, 2) + b) / gamma))
# k = t if t % 2 else t + 1
# print(k)
# # 32, 64 -- 3
# # 128, 256 -- 5



# net8 new, two efficient_CA_conv
class Efficient_DCSAF5(nn.Module):
    def __init__(self, dim_L, dim , num_heads):
        super(Efficient_DCSAF5, self).__init__()
        self.num_heads = num_heads
        self.trans_channel = nn.Conv2d(dim, dim_L, kernel_size=1)
        self.L_norm = LayerNorm2d(dim_L)
        self.H_norm = LayerNorm2d(dim_L)
        """fusion L and H"""
        self.L_q_conv = nn.Conv2d(dim_L, dim_L, kernel_size=1)
        self.L_q_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim_L)

        self.L_v_conv = nn.Conv2d(dim_L, dim_L, kernel_size=1)
        self.L_v_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim_L)

        self.H_k_conv = nn.Conv2d(dim_L, dim_L, kernel_size=1)
        self.H_k_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim_L)

        self.H_v_conv = nn.Conv2d(dim_L, dim_L, kernel_size=1)
        self.H_v_dwconv = nn.Conv2d(dim_L, dim_L, kernel_size=3, stride=1, padding=1, groups=dim_L)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.project_out = nn.Conv2d(dim_L, dim_L, kernel_size=1)

        # Efficient Channel Attention
        ch = int(dim/self.num_heads )
        cl = int(dim_L/self.num_heads)
        b=1
        gamma=2
        th = int(abs((math.log(ch, 2) + b) / gamma))
        tl = int(abs((math.log(cl, 2) + b) / gamma))
        kh = th if th % 2 else th + 1
        kl = tl if tl % 2 else tl + 1
        self.efficient_CA_conv_H = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=kh, padding=int(kh/2), bias=False)
        self.efficient_CA_conv_L = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=kl, padding=int(kl/2), bias=False)
        self.soft = nn.Softmax(dim = -1)

    def forward(self, lr_inp, hr_inp):
        hr_inp = self.trans_channel(hr_inp)
        b, c, h, w = lr_inp.shape
        lr = self.L_norm(lr_inp)
        hr = self.H_norm(hr_inp)

        """fusion lr and hr"""
        lr_q = self.L_q_dwconv(self.L_q_conv(lr))
        lr_v = self.L_v_dwconv(self.L_v_conv(lr))
        hr_k = self.H_k_dwconv(self.H_k_conv(hr))
        hr_v = self.H_v_dwconv(self.H_v_conv(hr))

        lr_q = rearrange(lr_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        lr_v = rearrange(lr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_k = rearrange(hr_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        hr_v = rearrange(hr_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)

        lr_q = torch.nn.functional.normalize(lr_q, dim=-1)
        hr_k = torch.nn.functional.normalize(hr_k, dim=-1)

        attn = (lr_q @ hr_k.transpose(-2, -1)) * self.temperature # b head (c/head) (c/head)

        attnh = self.efficient_CA_conv_H(attn) # b head (c/head) (c/head)
        attnl = self.efficient_CA_conv_L(attn) # b head (c/head) (c/head)
        temp = torch.cat((attnh, attnl), dim = 3)
        temp = self.soft(temp)
        attnL, attnH = torch.chunk(temp, 2, dim=3)

        out_H = (attnH @ hr_v)
        out_L = (attnL @ lr_v)

        out_H = rearrange(out_H, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_L = rearrange(out_L, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = out_H + out_L
        fusion = self.project_out(fusion) + lr_inp

        return fusion

###





















####################################################################
class GatedGuide(nn.Module):
    def __init__(self, dim, reduction=8):
        super(GatedGuide, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.soft = nn.Softmax(dim = 1)

        d = max(int(dim/reduction), 4)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, dim, 1, bias=False)
        )

    def forward(self, lr, hr):
        lr = self.conv(lr)
        hr = self.conv(hr)
        guide = self.mlp(lr)
        guide = self.soft(guide)

        out = hr + hr * guide
        return out




class new_GatedGuide(nn.Module):
    def __init__(self, dim, dim_L = 3):
        super(new_GatedGuide, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.conv_dw = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=3 // 2, groups=dim*2, padding_mode='reflect'),
            nn.GELU()
        )
        self.mask_in = nn.Sequential(
            nn.Conv2d(dim_L, self.dim_sp, 1),
            nn.GELU()
        )
        self.mask_dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp // 2, 1, kernel_size=3, padding=3 // 2, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.mask_dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp // 2, 1, kernel_size=5, padding=5 // 2, padding_mode='reflect'),
            nn.Sigmoid()
        )
        # self.mask_out = nn.Sequential(
        #     nn.Conv2d(2, dim_L * 2, 1),
        #     nn.GELU()
        # )

    def forward(self, lr, hr):
        residual = hr
        hr = self.conv_init(hr)
        hr = self.conv_dw(hr)
        hr = list(torch.split(hr, self.dim, dim=1))
        lr = self.mask_in(lr)
        lr = list(torch.split(lr, self.dim_sp//2, dim=1))
        lr[0] = self.mask_dw_conv_1(lr[0])
        lr[1] = self.mask_dw_conv_2(lr[1])
        hr[0] = lr[0] * hr[0]
        hr[1] = lr[1] * hr[1]
        hr = torch.cat(hr, dim=1)
        hr = self.conv_fina(hr) + residual
        # lr = self.mask_out(torch.cat(lr, dim=1))
        # print("lr",lr.shape)

        return hr



















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

if __name__ == '__main__':
    c = 256
    # q = torch.randn((2, c, 256, 256)).cuda()
    # k = torch.randn((2, c, 256, 256)).cuda()
    # v = torch.randn((2, c, 256, 256)).cuda()
    # net = GFM(dim = c,num_heads=8).cuda()
    # flops, params = profile(net, inputs=(q,k,v,))
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))
    c = 256
    head = 2
    
    # net = DCSAF(dim = c,num_heads = head)
    # lr = torch.randn(1, c, 960, 540)
    # hr = torch.randn(1, c, 960, 540)
    # flops, params = profile(net, inputs=(lr, hr))
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))
    # #  Number of parameters:0.4111 M


    # net = Efficient_DCSAF(dim = c,num_heads = head)
    # lr = torch.randn(1, c, 960, 540)
    # hr = torch.randn(1, c, 960, 540)
    # flops, params = profile(net, inputs=(lr, hr))
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))
    #  Number of parameters:0.4052 M

    # c2 = int((c*c)/(head*head))
    # c2 = int(c2 / 2)
    # net2 = SKFusion(dim = c2)
    # x = torch.randn(1, c2, head, 1)
    # y = torch.randn(1, c2, head, 1)
    # flops, params = profile(net2, inputs = [x,y])
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))

    # c = int(128)
    # h = int(960)
    # w = int(540)
    # net = GatedGuide(dim = c)
    # x = torch.randn(1, c, h, w)
    # y = torch.randn(1, c, h, w)
    # flops, params = profile(net, inputs=(x, y))
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))
    # # out = net(x, y)

    # 128 :0.1516 M  64 0.0379 M  32 :0.0095 M






    # net = new_GatedGuide(dim_L = 3, dim = 256)
    # x = torch.randn(1, 256, 960, 540)
    # mask = torch.randn(1, 3, 960, 540)
    # flops, params = profile(net, inputs=(mask, x))
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))



    lr = torch.randn(1, 32, 64, 64)
    hr = torch.randn(1, 256, 64, 64)
    net = Efficient_DCSAF6(32, 256, 2)
    flops, params = profile(net, inputs=(lr, hr))
    print(' Number of parameters:%.4f M' % (params / 1e6))
    print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))