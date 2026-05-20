import torch
import torch.nn as nn
from thop import profile
import os

import torch.nn.functional as F

try:
    from models.NAFBlock import NAFBlock
    from models.restormer_arch import TransformerBlock, Upsample, Downsample
    from models.GFM import GFM, CSA, SKFusion
    from models.g_Unet import BasicLayer
    from models import lr_scheduler as lr_scheduler
except:
    from NAFBlock import NAFBlock
    from restormer_arch import TransformerBlock, Upsample, Downsample  
    from GFM import GFM, CSA, SKFusion
    from g_Unet import BasicLayer
    import lr_scheduler as lr_scheduler

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from utils.loss import SSIM, PSNR
from utils.logger import get_root_logger


from models.net import Network as test_model

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = opt['device']                                                                                                                        
        self.net = test_model().to(self.device)
        # self.net = Baseline().to(self.device)
        self.optimizer = self.setup_optimizer()
        self.scheduler =self.setup_scheduler()


        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM()
        # self.psnr_loss = PSNR

    # 设置优化器
    def setup_optimizer(self):
        """Set up optimizer."""        
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optimizer'].pop('type')
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(optim_params, **train_opt['optimizer']) # Adam(self.net.parameters(), lr=cfg.lr)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(optim_params, **train_opt['optimizer'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        return optimizer

    # 设置学习率调度器
    def setup_scheduler(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            scheduler = lr_scheduler.MultiStepLR(self.optimizer, **train_opt['scheduler'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            scheduler = lr_scheduler.CosineAnnealingRestartLR(self.optimizer, **train_opt['scheduler'])
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            scheduler = lr_scheduler.CosineAnnealingWarmupRestarts(self.optimizer, **train_opt['scheduler'])
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            scheduler = lr_scheduler.CosineAnnealingRestartCyclicLR(self.optimizer, **train_opt['scheduler'])
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **train_opt['scheduler'])
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            print('..', 'CosineAnnealingLR_With_Restart')
            scheduler = lr_scheduler.CosineAnnealingLRWithRestart(self.optimizer, **train_opt['scheduler'])
        elif scheduler_type == 'LinearLR':
            scheduler = lr_scheduler.LinearLR(self.optimizer, train_opt['total_iter'])
        elif scheduler_type == 'VibrateLR':
            scheduler = lr_scheduler.VibrateLR(self.optimizer, train_opt['total_iter'])
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
        # print("------------------------------------schedulers:", scheduler_type)
        return scheduler

    # 保存模型
    def save_network(self, name, current_epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        state = {
            'net': self.net.state_dict(),
            'epoch': current_epoch,
            'iter': current_iter,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        if name == 'best':
            save_filename = f'{name}_{current_epoch}.pth'
        else:
            save_filename = f'{current_epoch}.pth'
        save_path = os.path.join(self.opt['path']['experiments_root'] + self.opt['datasets']['train']['type'], save_filename)
        torch.save(state, save_path)
        
    # 保存模型
    def save_network2(self, name, current_epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        state = {
            'net': self.net,
            'epoch': current_epoch,
            'iter': current_iter,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler
        }
        if name == 'fullnet':
            save_filename = f'{name}_{current_epoch}.pth'
        else:
            save_filename = f'{current_epoch}.pth'
        save_path = os.path.join(self.opt['path']['experiments_root'] + self.opt['datasets']['train']['type'], save_filename)
        torch.save(state, save_path)
        

    # 加载模型
    def resume_network(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        self.net.load_state_dict(resume_state['net'])
        for i in range(1,resume_state['epoch'] + 1):
            self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']     

    def train_batch(self, batch_train):
        X, Y = batch_train['X'].to(self.device), batch_train['Y'].to(self.device)

        # g_derain,derain,refine = self.net(X)
        g_derain,derain = self.net(X)

        self.optimizer.zero_grad()

        """backward"""
        g_Y = torch.nn.functional.interpolate(Y, size=(Y.shape[2] // 4, Y.shape[3] // 4), mode='bilinear', align_corners=True)
        l1_loss1 = self.l1_loss(g_derain, g_Y)
        l1_loss2 = self.l1_loss(derain, Y)
        ssim = self.ssim_loss(derain, Y)

        loss = 0.5 * l1_loss1 + l1_loss2 + 0.2*(1 - ssim)

        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy(), ssim.detach().cpu().numpy(), 0.5 * l1_loss1.detach().cpu().numpy(), l1_loss2.detach().cpu().numpy(), 0.2*(1 - ssim.detach().cpu().numpy())   ####################

    def val_batch(self, batch_val):
        X, Y = batch_val['X'].to(self.device), batch_val['Y'].to(self.device)
        padh = int(batch_val['padh'])
        padw = int(batch_val['padw'])

        with torch.no_grad():
            _,derain = self.net(X)
            b, c, h, w = derain.shape
            derain = derain[: , : , 0:h - padh , 0:w - padw]

        ssim = self.ssim_loss(derain, Y)
        psnr = PSNR(derain.data.cpu().numpy()*255, Y.data.cpu().numpy()*255)

        return ssim.detach().cpu().numpy(), psnr

    def update_learning_rate(self, epoch, current_iter, warmup_iter = -1):
        if epoch > 0:
            self.scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)  


# if __name__ == '__main__':
# 	x = torch.randn((1, 3, 256, 256))
# 	net = Baseline()
# 	flops, params = profile(net, inputs=(x,))
# 	print(' Number of parameters:%.4f M' % (params / 1e6))
# 	print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))