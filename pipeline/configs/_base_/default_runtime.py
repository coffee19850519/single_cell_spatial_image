# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/mnt/mmseg_singlecell/work_dirs/1v_deeplabv3_r101-d8_512x1024_80k_singlecell/latest.pth'
# resume_from = '/mnt/mmseg_singlecell/work_dirs/1v_deeplabv3_r101-d8_512x1024_80k_singlecell/latest.pth'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
