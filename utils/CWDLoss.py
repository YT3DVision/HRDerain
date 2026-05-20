import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..builder import LOSSES



# def attention_transform(feat):
#     return F.normalize(feat.pow(2).mean(1).view(feat.size(0), -1))


# def similarity_transform(feat):
#     feat = feat.view(feat.size(0), -1)
#     gram = feat @ feat.t()
#     return F.normalize(gram)


# _TRANS_FUNC = {"attention": attention_transform, "similarity": similarity_transform, "linear": lambda x : x}


def ChannelWiseDivergence(feat_t, feat_s):
    assert feat_s.shape[-2:] == feat_t.shape[-2:]
    N, C, H, W = feat_s.shape
    softmax_pred_T = F.softmax(feat_t.reshape(-1, W * H) / 4.0, dim=1)
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    loss = torch.sum(softmax_pred_T *
                     logsoftmax(feat_t.reshape(-1, W * H) / 4.0) -
                     softmax_pred_T *
                     logsoftmax(feat_s.reshape(-1, W * H) / 4.0)) * (
                         (4.0)**2)
    loss =  loss / (C * N)   
    return loss


# @LOSSES.register_module()
class AlignmentLoss(nn.Module):

    def __init__(self, 
                loss_weight=1.0,
                loss_name='loss_guidance',
                inter_transform_type='linear'):
        super(AlignmentLoss, self).__init__()
        self.inter_transform_type=inter_transform_type
        self._loss_name = loss_name
        self.loss_weight = loss_weight
    
    def forward(self, x_guidance_feature):
        loss_inter = x_guidance_feature[0][0].new_tensor(0.0)  
        # for i in range(4):
        #     feat_t = x_guidance_feature[0][i]
        #     feat_s = x_guidance_feature[1][i]
        #     if feat_t.size(-2)!=feat_s.size(-2) or feat_t.size(-1)!=feat_s.size(-1):
        #         dsize = (max(feat_t.size(-2), feat_s.size(-2)), max(feat_t.size(-1), feat_s.size(-1)))
        #         #feat_t = F.interpolate(feat_t, dsize, mode='bilinear', align_corners=False)
        #         feat_s = F.interpolate(feat_s, dsize, mode='bilinear', align_corners=False)
        #     # loss_inter = loss_inter + self.loss_weight[i]*ChannelWiseDivergence(feat_t, feat_s)
        #     loss_inter = loss_inter + self.loss_weight*ChannelWiseDivergence(feat_t, feat_s)
        feat_t = x_guidance_feature[0]
        feat_s = x_guidance_feature[1]
        if feat_t.size(-2)!=feat_s.size(-2) or feat_t.size(-1)!=feat_s.size(-1):
            dsize = (max(feat_t.size(-2), feat_s.size(-2)), max(feat_t.size(-1), feat_s.size(-1)))
            #feat_t = F.interpolate(feat_t, dsize, mode='bilinear', align_corners=False)
            feat_s = F.interpolate(feat_s, dsize, mode='bilinear', align_corners=False)
        loss_inter = loss_inter + self.loss_weight*ChannelWiseDivergence(feat_t, feat_s)       
        return loss_inter
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    


if __name__ == "__main__":
    # 创建 AlignmentLoss 实例
    alignment_loss = AlignmentLoss()

    # 准备输入特征
    # x_guidance_feature = [
    #     [torch.randn(1, 64, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 64, 8, 8), torch.randn(1, 64, 4, 4)],
    #     [torch.randn(1, 64, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 64, 8, 8), torch.randn(1, 64, 4, 4)]
    # ]
    x = torch.randn(1, 63, 32, 32)
    y1 = torch.randn(1, 1, 32, 32)
    y2 = torch.randn(1, 1, 32, 32)
    z1 = torch.cat((x, y1), 1)
    z2 = torch.cat((x, y2), 1)
    x_guidance_feature = [torch.randn(1, 64, 32, 32), torch.randn(1, 64, 32, 32)]
    x_guidance_feature2 = [z1, z2]

    # # 计算损失
    loss1 = alignment_loss(x_guidance_feature)
    loss2 = alignment_loss(x_guidance_feature2)
    print("Loss1:", loss1.item())
    print("Loss2:", loss2.item())

    # 进行反向传播
    # loss.backward()
