import os
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils.options as option
from utils.logger import get_root_logger, get_env_info
from data import create_dataset, create_val_dataset
from models import create_model


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
    print("显存占用率: {:.2%}".format( info.used / info.total), end="    ")
    nvmlShutdown()


# options
def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="para.yml" , help="Path to option YMAL file.")
    # parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher') # 分布式
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # random seed 为CPU\GPU设置种子，保证每次的随机初始化都是相同的，从而保证结果可以复现。"""
    seed = opt.get('manual_seed')
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    return opt

def init_loggers(opt):
    log_dir = opt['path']['log']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(opt['path']['log'], 
                            f"{opt['datasets']['train']['type']}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


# 创建数据加载器
def load_train_data(opt, gt_size, batch_size):
    train_loader = None
    dataset_opt = opt['datasets']['train'].copy()
    # dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
    train_set = create_dataset(dataset_opt, gt_size)
    # train_loader = create_dataloader(train_set, dataset_opt, batch_size)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, 
                              num_workers = opt['num_workers'], prefetch_factor = opt['prefetch_factor'], drop_last=True)

    # logger.info(f'\n\tNumber of train images: {len(train_set)}')
    return train_loader

def load_val_data(opt):
    val_loader = None
    dataset_opt = opt['datasets']['val'].copy()
    val_set = create_val_dataset(dataset_opt)
    val_loader  = DataLoader(
        val_set,batch_size = dataset_opt['BatchSize'], shuffle=True, num_workers = opt['num_workers'], prefetch_factor = opt['prefetch_factor'], drop_last=False)
    
    # logger.info(f'\n\tNumber of val images/folders in {dataset_opt["type"]}: {len(val_set)}')
    return val_loader

# 创建路径
def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + time.strftime('%Y%m%d_%H%M%S', time.localtime())
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)

def main():
    opt = parse_options(is_train=True)
    print(opt['device'])
    # print(opt['num_workers'])

    # initialize loggers
    logger = init_loggers(opt)

    # automatic resume
    state_folder_path = opt['path']['training_states'] + opt['datasets']['train']['type']
    train_states = []
    try:
        train_states = os.listdir(state_folder_path)
    except:
        train_states = []

    # load resume states if necessary
    resume_state = None
    if (len(train_states) > 0 and opt['train']['train_resume']):
        best_files = [x for x in train_states if x.startswith('best_')]
        max_num = max([int(x[5:-4]) for x in best_files])
        max_state_file = 'best_{}.pth'.format(max_num)
        resume_state_path = os.path.join(state_folder_path, max_state_file)
        if opt['device']:
            device_id = opt['device']
        else:
            device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None
        if opt['is_train']:
            path_opt = opt['path']['experiments_root'] + opt['datasets']['train']['type']
            mkdir_and_rename(path_opt)


    # create model
    if resume_state:  # resume training
        # check_resume(opt, resume_state['iter'])
        model = create_model(opt, logger)
        model.resume_network(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "f"iter: {resume_state['iter']}.")
        print(f"Resuming training from epoch: {resume_state['epoch']}, "f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch'] + 1
        current_iter = resume_state['iter'] + 1
    else:
        model = create_model(opt, logger)
        start_epoch = 0
        current_iter = 0


    # start_training_info
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    print(f'Start training from epoch: {start_epoch}, iter: {current_iter}')

    # 阶段式训练参数
    epochs = opt['datasets']['train'].get('epochs')
    batch_sizes = opt['datasets']['train'].get('batch_sizes')
    gt_sizes = opt['datasets']['train'].get('gt_sizes')

    # 计算每个阶段的累积迭代次数
    groups = np.array([sum(epochs[0:i + 1]) for i in range(0, len(epochs))])

    # 用于记录每个阶段时的日志信息
    logger_j = [True] * len(groups)

    # Tensorboard可视化 # tensorboard --logdir=EXP/tb_logger
    tensorboard_dir = os.path.join(opt['path']['tb_logger'], opt['datasets']['train']['type'])
    tb_writer = SummaryWriter(log_dir = tensorboard_dir)
    tags = ['Train_L1', 'Train_SSIM', 'Eval_PSNR', 'Eval_SSIM', 'lr']

    # scale = opt['scale']
    epoch = start_epoch
    total_epoch = opt['train']['total_epoch']

    logger.info('\nUsing datasets:{} \n'.format(opt['datasets']['train']['type']))
    print('\nUsing datasets:{} \n'.format(opt['datasets']['train']['type']))
    val_loader = load_val_data(opt)
    best_epoch = 0
    best_psnr = 0
    best_ssim = 0


    for epoch in range(epoch, total_epoch+1):
        j = ((epoch>groups) !=True).nonzero()[0]
        if len(j) == 0:
            i = len(groups) - 1
        else:
            i = j[0]
        gt_size = gt_sizes[i]
        batch_size = batch_sizes[i]
        # 加载数据集
        train_loader = load_train_data(opt, gt_size, batch_size)

        if logger_j[i]:
            logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(gt_size, batch_size))
            print('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(gt_size, batch_size))
            logger_j[i] = False

        time_start = time.time()
        l1_all = 0
        SSIM_all = 0
        train_sample = 0    
        model.net.train()
        loss1_all =0 ####################
        loss2_all = 0 ####################
        ####
        # printGPU()
        ####


        for _, batch_train in enumerate(tqdm(train_loader, ncols=100), 0):
            current_iter += 1
            # 更新参数，计算loss
            l1_loss, ssim, loss1, loss2 = model.train_batch(batch_train) ####################
            l1_all += l1_loss
            SSIM_all += ssim
            train_sample += 1
            loss1_all += loss1 ####################
            loss2_all += loss2 ####################
        
        SSIM = SSIM_all / train_sample
        l1_all  = l1_all * train_loader.batch_size
        loss1_all = loss1_all * train_loader.batch_size ####################
        loss2_all = loss2_all * train_loader.batch_size ####################
        print("loss1_all: {:.4f} loss2_all: {:.4f}".format(loss1_all, loss2_all)) ####################

        """Evaluation"""

        if epoch % opt['val']['val_freq'] == 0:           
            ssim_val = []
            psnr_val = []
            model.net.eval()
            
            for _, batch_val in enumerate(tqdm(val_loader, ncols=100), 0):
            # for iii, batch_val in enumerate(val_loader, 0):
                
                # ####
                # printGPU()
                # ####
                # X = batch_val['X']
                # print(X.size())
                # print(iii)

                ssim, psnr = model.val_batch(batch_val)
                ssim_val.append(ssim)
                psnr_val.append(psnr)


            # ssim_val = torch.stack(ssim_val).mean().item()
            ssim_val = np.stack(ssim_val).mean().item()
            psnr_val = np.stack(psnr_val).mean().item()

            tb_writer.add_scalar(tags[2],psnr_val,epoch)
            tb_writer.add_scalar(tags[3],ssim_val,epoch)



            #如果psnr_avg大于best_psnr则单独保存
            if(psnr_val >= best_psnr):
                best_psnr = psnr_val
                best_ssim = ssim_val
                best_epoch = epoch

                model.save_network('best', epoch, current_iter)
                if epoch > 800:
                  model.save_network2('fullnet', epoch, current_iter)
            
            # 输出最好的结果  log
            logger.info(f"best_epoch: {best_epoch}, best_psnr: {best_psnr}, best_ssim: {best_ssim}")
            print("[epoch %d PSNR: %.4f SSIM:%.4f --- best_epoch %d Best_PSNR %.4f Best_SSIM %.4f]" % 
                  (epoch, psnr_val,ssim_val ,best_epoch, best_psnr,best_ssim))
        
        model.update_learning_rate(epoch, current_iter, warmup_iter = opt['train'].get('warmup_iter', -1))

        logger.info(f"Epoch: {epoch}, Iteration: {current_iter}")
        logger.info(f"Learning Rates: {model.optimizer.param_groups[0]['lr']}")
        # logger.info(f"epoch Time: {time.time() - time_start}")
        logger.info(f"Loss: {l1_all}, SSIM: {SSIM}")
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tSSIM: {:.4f}\tBase_lr {:.8f}"
              .format(epoch,time.time() - time_start, l1_all, SSIM, model.scheduler.get_last_lr()[0]))


        print("------------------------------------------------------------------")
        # if epoch == 1000000000000000000:
        #     model.save_network('model_latest', epoch, current_iter)
        tb_writer.add_scalar(tags[0], l1_all, epoch)
        tb_writer.add_scalar(tags[1], SSIM, epoch)
        tb_writer.add_scalar(tags[4], model.scheduler.get_last_lr()[0], epoch) #model.optimizer.param_groups[0]['lr']

if __name__ == '__main__':
    main()
