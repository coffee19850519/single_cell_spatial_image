import os
import cv2
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def case_study(path,r_tuple,g_tuple,b_tuple):

    new_path = path.replace('pseudo_images','case_study')
    # print(new_path)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for name in os.listdir(path):
         
        img = cv2.imread(path+name)
        img = cv2.resize(img, (600, 600), cv2.INTER_NEAREST)

        r_min = r_tuple[0]
        r_max = r_tuple[1]
        g_min = g_tuple[0]
        g_max = g_tuple[1]
        b_min = b_tuple[0]
        b_max = b_tuple[1]

        res = img.shape[0]
        img_filter = np.ones(shape=(600, 600, 3), dtype=np.uint8) * 255
        for i in range(res):
            for j in range(res): #[137 115 107]
                if (img[i][j][0] >= r_min  and img[i][j][0] <= r_max) \
                and (img[i][j][1] >= g_min and img[i][j][1] <= g_max) \
                and (img[i][j][2] >= b_min and img[i][j][2] <= b_max):
                    img_filter[i][j] = img[i][j]

        cv2.imwrite(new_path + 'case_study_' + name, img_filter)


# r_tuple = (1,115)
# g_tuple = (0,96)
# b_tuple = (0,74)
