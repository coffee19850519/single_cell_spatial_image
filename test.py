import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import pandas as pd
import cv2
from PIL import Image
import json
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import sys
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=0)


def segmentation(adata,img_path,label_path,method,checkpoint_path,device):  

    if device=='cpu':
        print('cpu function is under development...')
        sys.exit(0)

    config = './configs/deeplabv3_r101-d8_512x512_80k_singlecell.py'
    checkpoint = checkpoint_path
    if label_path == None:
        output_folder = img_path.split('/')[0]+'/segmentation_test/'
    else:
        output_folder = img_path.split('/')[0] + '/segmentation_evaluation/'
    show_dir = output_folder+'show_temp/'
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)
    cfg = mmcv.Config.fromfile(config)
    cfg.data.test['data_root'] = None
    cfg.data.test['img_dir'] = img_path
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True


    # init distributed env first, since logger depends on the dist info.
    launcher = 'none'
    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)

    dataset = build_dataset(cfg.data.test)


    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)


    # build the model and load checkpoint
    cfg.model.train_cfg = None

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']
    
    efficient_test = True
    # if args.eval_options is not None:
    #     efficient_test = args.eval_options.get('efficient_test', False)
    show = None
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs, top1_csv_name = single_gpu_test(adata, model, data_loader,label_path, output_folder, show, show_dir,
                                  efficient_test)
    return top1_csv_name
        

if __name__ == '__main__':
    main()
