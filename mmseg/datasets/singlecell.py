import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SCDataset(CustomDataset):
    """HRF dataset.

    In segmentation map annotation for HRF, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    # CLASSES = ('0_dark_red', '1_light_red',
    #            '2_dark_orange', '3_light_orange',
    #            '4_dark_yellow', '5_light_yellow',
    #            '6_dark_green', '7_light_green',
    #            '8_dark_cyan', '9_light_cyan',
    #            '10_dark_blue','11_light_blue',
    #            '12_dark_purple', '13_light_purple')
    #
    # PALETTE = [[197, 0, 35], [245, 168, 154],
    #            [236,135,140], [250,206,156],
    #            [249,244,0], [254,248,134],
    #            [72,150,32], [200,226,177],
    #            [0,132,137], [153,209,211],
    #            [24,71,133], [148,170,214],
    #            [73,7,97], [170,135,184]]

    CLASSES = (
               'background',
               'Layer1',
               'Layer2',
               'Layer3',
               'Layer4',
               'Layer5',
               'Layer6',
               'WM'
               )

    PALETTE = [
               [255,255,255],
               [0,0,255],
               [0,255,00],
               [255,0,0],
               [255,0,255],
               [0,255,255],
               [255,255,0],
               [0,0,0]
               ]

    def __init__(self, **kwargs):
        super(SCDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            # ignore_index=0,
            # reduce_zero_label=False,
            ignore_index=0,
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
