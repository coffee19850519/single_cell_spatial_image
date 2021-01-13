import pandas as pd
import umap
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import permutations
from tkinter import *
import tkinter.messagebox as messagebox

def scale_to_RGB(channel,truncated_percent):
    truncated_down = np.percentile(channel, truncated_percent)
    truncated_up = np.percentile(channel, 100 - truncated_percent)
    channel_new = (255 / (truncated_up - truncated_down)) * channel
    channel_new[channel_new < 0] = 0
    channel_new[channel_new > 255] = 255
    return np.uint8(channel_new)

def correlation_between_images(X_RGB, transformed_RGB):
    best_pcc = 0
    best_perm = (0,1,2)
    for perm in permutations(range(3), 3):
        current_pcc, p_value = pearsonr(X_RGB.reshape((-1)), transformed_RGB[:, perm].reshape((-1)))
        if abs(current_pcc) > best_pcc:
            best_pcc = abs(current_pcc)
            best_perm = perm
    return  best_pcc, best_perm

def nan_check_in_datframe():
    pass


def transform_expression_to_RGB(expression_file):
    data = pd.read_csv(expression_file)

    X = data.loc[data['in_tissue'] != 0]

    # X.iloc[[199,203, 205, 2793, 2575, 2578, 2580, 2582], :].to_csv(expression_file[:-4] + '_nan.csv',index = False)

    # save_spot_RGB_to_image(X, expression_file)

    values = X.iloc[:, 9:].values
    transformer = umap.UMAP(n_neighbors= 10, min_dist=0.2, n_components=3)
    try:
        X_transformed = transformer.fit(values).transform(values)
    except Exception as e:
        print(str(e))
        return

    best_pcc = 0
    best_perm = (0,1,2)
    best_percentile_0 = 0
    best_percentile_1 = 0
    best_percentile_2 = 0
    #do grid search for the percentile parameters
    for percentile_0 in range(0, 50):
        for percentile_1 in range(0, 50):
            for percentile_2 in range(0, 50):
                # plot_box_for_RGB(X_transformed)
                temp_X_transform = np.zeros_like(X_transformed, np.int8)
                temp_X_transform[:, 0] = scale_to_RGB(X_transformed[:, 0], percentile_0)
                temp_X_transform[:, 1] = scale_to_RGB(X_transformed[:, 1], percentile_1)
                temp_X_transform[:, 2] = scale_to_RGB(X_transformed[:, 2], percentile_2)

                pcc, perm = correlation_between_images(X.iloc[:, 6:9].values, X_transformed)
                if best_pcc < pcc:
                    best_pcc = pcc
                    best_perm = perm
                    best_percentile_0 = percentile_0
                    best_percentile_1 = percentile_1
                    best_percentile_2 = percentile_2
                del temp_X_transform

    #re-produce optimal RGB projection result
    X_transformed[:, 0] = scale_to_RGB(X_transformed[:, 0], best_percentile_0)
    X_transformed[:, 1] = scale_to_RGB(X_transformed[:, 1], best_percentile_1)
    X_transformed[:, 2] = scale_to_RGB(X_transformed[:, 2], best_percentile_2)

    print('the best pcc for the file {:s} is {:.3f}'.format(expression_file,best_pcc))

    save_transformed_RGB_to_image_and_csv(X, X_transformed[:, best_perm], expression_file)



    del X_transformed, data, X, values, transformer




def contour_detection(img, mask):

    # imputed_img = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

    #GaussianBlur
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)

    # Grayscale
    gray_img = cv2.cvtColor(blurred_img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    # Find Canny edges
    edge_img = cv2.Canny(gray_img.astype(np.uint8), 30, 100)
    # Finding Contours
    contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # Draw all contours
    # -1 signifies drawing all contours
    # cv2.drawContours(imputed_img, contours, -1, (0, 0, 0), 1)
    # cv2.imwrite(os.path.splitext(expression_file)[0] + '_contour.jpg', imputed_img)

    del blurred_img, gray_img, edge_img, mask

    return contours, hierarchy

def plot_box_for_RGB(RGB_data):
    fig = plt.figure()
    plt.boxplot(RGB_data)
    plt.xticks([1,2,3], ['R\'','G\'','B\''],  rotation=0)
    plt.show()

def save_transformed_RGB_to_image_and_csv(X, X_transformed, data_file_name):

    # full size spot number
    spot_row = X.loc[:, 'pxl_row_in_fullres'].values
    spot_col = X.loc[:, 'pxl_col_in_fullres'].values

    max_row = np.max(spot_row)
    max_col = np.max(spot_col)

    img = np.ones(shape=(max_row, max_col, 3), dtype= np.uint8) * 255

    # mask = np.ones(shape=(max_row + 1, max_col + 1), dtype=np.uint8)
    # should follow the spot permutation at full resolution image
    for index in range(len(X_transformed)):
        # plot dot according to RGB, ra
        # img[spot_row[index], spot_col[index]] = np.uint8(X_transformed[index])

        cv2.circle(img, (spot_row[index], spot_col[index]), radius= 65,
                   color= (int(X_transformed[index][0]),int(X_transformed[index][1]),int(X_transformed[index][2])),
                   thickness= -1)
        # mask[spot_row[index], spot_col[index]] = 0

    # cv2.imwrite(r'/home/fei/Desktop/1.1.0/image.jpg', img)
    # img = cv2.imread(r'/home/fei/Desktop/1.1.0/image.jpg')

    # cv2.imwrite(r'/home/fei/Desktop/1.1.0/mask.jpg', mask)
    # del mask
    # mask = cv2.imread(r'/home/fei/Desktop/1.1.0/mask.jpg')
    # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # resize_img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # inpaint_img = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
    cv2.imwrite(os.path.splitext(data_file_name)[0] + '_transformed.jpg', img)
    # cv2.imwrite(os.path.splitext(data_file_name)[0] + '_inpaint.jpg', inpaint_img)
    # contours, hirarchy = contour_detection(img, None)
    # # draw the conrours on img and then resize and save
    # cv2.drawContours(img, contours, -1, (0, 0, 0), 1)
    # resize_contour = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.splitext(data_file_name)[0] + '_contour.jpg', resize_contour)


    # save transformed_X back to csv file
    X.insert(9, 'transformed_R',X_transformed[:, 0], True)
    X.insert(10, 'transformed_G',X_transformed[:, 1], True)
    X.insert(11, 'transformed_B',X_transformed[:, 2], True)

    # #then save to csv file
    X.iloc[:, 0:12].to_csv(os.path.splitext(data_file_name)[0] + '_newRGB.csv', index = False)

    del img, spot_row, spot_col#, mask, inpaint_img,  resize_contour#resize_img,



def save_spot_RGB_to_image(X, data_file ):
    # data = pd.read_csv(data_file, sep='\t')
    # X = data.loc[data['in_tissue'] != 0]
    values = X.iloc[:,6:9].values
    spot_row = X.iloc[:, 2].values
    spot_col = X.iloc[:, 3].values

    max_row = np.int(np.max(spot_row))
    max_col = np.int(np.max(spot_col))

    img = np.ones(shape=(max_row+1, max_col+1,  3), dtype= np.float) * 255


    for index in range(len(values)):
        img[spot_row[index], spot_col[index]] = values[index]

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resize_img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.splitext(data_file)[0] + '_raw.jpg', resize_img)

    del values, spot_row, max_col, img, resize_img


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.nameInput = Entry(self)
        self.nameInput.pack()
        self.alertButton = Button(self, text='Hello', command=self.hello)
        self.alertButton.pack()

    def hello(self):
        name = self.nameInput.get() or 'world'
        messagebox.showinfo('Message', 'Hello, %s' % name)


# app = Application()
# # 设置窗口标题:
# app.master.title('Hello World')
# # 主消息循环:
# app.mainloop()


data_folder = r'/home/fei/Desktop/test/expression/'
for expression_file in os.listdir(data_folder):
    if os.path.splitext(expression_file)[1] == '.csv':
        transform_expression_to_RGB(os.path.join(data_folder, expression_file))
        # save_spot_RGB_to_image(os.path.join(data_folder, expression_file))