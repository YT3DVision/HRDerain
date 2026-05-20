# import numpy as np
# import random
# import torch
# import torch.utils.data
# from functools import partial

# from utils.logger import get_root_logger


# 训练集
def create_dataset(dataset_opt, gt_size):
    type = dataset_opt['type']
    if type == 'RealRain1K_H':
        from data.Data_RealRain1K_H import Train_RealRain1K_H_Aug_Dataset
        set_opt = dataset_opt['RealRain1K_H'].copy()
        train_set = Train_RealRain1K_H_Aug_Dataset(set_opt, gt_size)
    elif type == 'RealRain1K_L':
        from data.Data_RealRain1K_L import Train_RealRain1K_L_Aug_Dataset
        set_opt = dataset_opt['RealRain1K_L'].copy()
        train_set = Train_RealRain1K_L_Aug_Dataset(set_opt, gt_size)
    elif type == 'LHP_Rain':
        from data.Data_LHP_Rain import Train_LHP_Rain
        set_opt = dataset_opt['LHP_Rain'].copy()
        train_set = Train_LHP_Rain(set_opt, gt_size)
    elif type == 'RainDirection':
        from data.Data_RainDirection import Train_RainDirection
        set_opt = dataset_opt['RainDirection'].copy()
        train_set = Train_RainDirection(set_opt, gt_size)
    else:
        raise ValueError(f'Dataset {type} is not found. train')
    
    return train_set

# def create_dataloader(train_set, dataset_opt, batch_size):
#     pass


# 验证集
def create_val_dataset(dataset_opt):
    type = dataset_opt['type']
    if type == 'RealRain1K_H':
        from data.Data_RealRain1K_H import Test_RealRain1K_H_Dataset_whole_resolution
        set_opt = dataset_opt['RealRain1K_H'].copy()
        val_set = Test_RealRain1K_H_Dataset_whole_resolution(set_opt)
    elif type == 'RealRain1K_L':
        from data.Data_RealRain1K_L import Test_RealRain1K_L_Dataset_whole_resolution
        set_opt = dataset_opt['RealRain1K_L'].copy()
        val_set = Test_RealRain1K_L_Dataset_whole_resolution(set_opt)
    elif type == 'LHP_Rain':
        from data.Data_LHP_Rain import Test_LHP_Rain ############################################！！！！！！！！！！！！！！！！！！
        set_opt = dataset_opt['LHP_Rain'].copy()
        val_set = Test_LHP_Rain(set_opt)
    elif type == 'RainDirection':
        from data.Data_RainDirection import Test_RainDirection ############################################！！！！！！！！！！！！！！！！！！
        set_opt = dataset_opt['RainDirection'].copy()
        val_set = Test_RainDirection(set_opt)
    else:
        raise ValueError(f'Dataset {type} is not found. val')

    return val_set

# 测试集
def create_test_dataset(dataset_opt):
    type = dataset_opt['type']
    if type == 'RealRain1K_H':
        from data.Data_RealRain1K_H import Test_RealRain1K_H_Dataset_whole_resolution
        set_opt = dataset_opt['RealRain1K_H'].copy()
        test_set = Test_RealRain1K_H_Dataset_whole_resolution(set_opt)
    elif type == 'RealRain1K_L':
        from data.Data_RealRain1K_L import Test_RealRain1K_L_Dataset_whole_resolution
        set_opt = dataset_opt['RealRain1K_L'].copy()
        test_set = Test_RealRain1K_L_Dataset_whole_resolution(set_opt)
    elif type == 'LHP_Rain':
        from data.Data_LHP_Rain import Test_LHP_Rain
        set_opt = dataset_opt['LHP_Rain'].copy()
        test_set = Test_LHP_Rain(set_opt)
    elif type == 'RainDirection':
        from data.Data_RainDirection import Test_RainDirection
        set_opt = dataset_opt['RainDirection'].copy()
        test_set = Test_RainDirection(set_opt)
    else:
        raise ValueError(f'Dataset {type} is not found. test')

    return test_set










# def  create_dataset(dataset_opt):
#     dataset_type = dataset_opt['type']
#     if dataset_type == 'Dataset_PairedImage':
#         from data.paired_image_dataset import Dataset_PairedImage
#         dataset = Dataset_PairedImage(dataset_opt)
#     else:
#         raise ValueError(f'Dataset {dataset_type} is not found.')
    
#     logger = get_root_logger()
#     logger.info(
#         f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} '
#         'is created.')
#     return dataset




# def create_dataloader(dataset,
#                       dataset_opt,
#                       num_gpu=1,
#                       dist=False,
#                       sampler=None,
#                       seed=None):
#     """Create dataloader.

#     Args:
#         dataset (torch.utils.data.Dataset): Dataset.
#         dataset_opt (dict): Dataset options. It contains the following keys:
#             phase (str): 'train' or 'val'.
#             num_worker_per_gpu (int): Number of workers for each GPU.
#             batch_size_per_gpu (int): Training batch size for each GPU.
#         num_gpu (int): Number of GPUs. Used only in the train phase.
#             Default: 1.
#         dist (bool): Whether in distributed training. Used only in the train
#             phase. Default: False.
#         sampler (torch.utils.data.sampler): Data sampler. Default: None.
#         seed (int | None): Seed. Default: None
#     """
#     phase = dataset_opt['phase']
#     rank, _ = get_dist_info()
#     if phase == 'train':
#         if dist:  # distributed training
#             batch_size = dataset_opt['batch_size_per_gpu']
#             num_workers = dataset_opt['num_worker_per_gpu']
#         else:  # non-distributed training
#             multiplier = 1 if num_gpu == 0 else num_gpu
#             batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
#             num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
#         dataloader_args = dict(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             sampler=sampler,
#             drop_last=True)
#         if sampler is None:
#             dataloader_args['shuffle'] = True
#         dataloader_args['worker_init_fn'] = partial(
#             worker_init_fn, num_workers=num_workers, rank=rank,
#             seed=seed) if seed is not None else None
#     elif phase in ['val', 'test']:  # validation
#         dataloader_args = dict(
#             dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
#     else:
#         raise ValueError(f'Wrong dataset phase: {phase}. '
#                          "Supported ones are 'train', 'val' and 'test'.")

#     dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

#     prefetch_mode = dataset_opt.get('prefetch_mode')
#     if prefetch_mode == 'cpu':  # CPUPrefetcher
#         num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
#         logger = get_root_logger()
#         logger.info(f'Use {prefetch_mode} prefetch dataloader: '
#                     f'num_prefetch_queue = {num_prefetch_queue}')
#         return PrefetchDataLoader(
#             num_prefetch_queue=num_prefetch_queue, **dataloader_args)
#     else:
#         # prefetch_mode=None: Normal dataloader
#         # prefetch_mode='cuda': dataloader for CUDAPrefetcher
#         return torch.utils.data.DataLoader(**dataloader_args)


# def worker_init_fn(worker_id, num_workers, rank, seed):
#     # Set the worker seed to num_workers * rank + worker_id + seed
#     worker_seed = num_workers * rank + worker_id + seed
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
