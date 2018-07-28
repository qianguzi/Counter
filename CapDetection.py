import os

import cv2
import numpy as np


class CapDetection():
    """ 
    利用霍夫圆检测得到候选区域. 
    """

    def __init__(self, imread_dir, imwrite_dir):
        """
        imread_dir: 输入图像路径
        imwrite_dir: 保存候选区域路径
        """
        self.imread_dir = imread_dir
        self.imwrite_dir = imwrite_dir
        self.imgs = self.read(imread_dir=imread_dir)

    def read(self, imread_dir):
        return {i: cv2.imread(imread_dir+i, 0) for i in os.listdir(imread_dir)}

    def save(self, can_imgs, imwrite_dir, img_name):
        """
        保存候选区域...
        """
        for n, i in enumerate(can_imgs):
            cv2.imwrite(imwrite_dir + img_name + str(n) + '.jpg', i)
        print(img_name + '的候选区域写入成功...')

    def hough(self, img, minDist, pre, minRadius=15, maxRadius=35):
        """
        获取感兴趣圆心...
        img: input image      minDist: 两个圆心之间的最小距离
        pre: 精度阈值          minRadius: 最小半径
        maxRadius: 最大半径
        """
        cir = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist, param1=100,
                               param2=pre, minRadius=minRadius, maxRadius=maxRadius)
        cir = np.uint16(np.round(cir))
        return cir[0]

    def preprocessing(self, img):
        """
        图像预处理...
        """
        pre_img = cv2.equalizeHist(img)
        pre_img = cv2.GaussianBlur(pre_img, (5, 5), 0)
        pre_img = cv2.Laplacian(pre_img, -1, ksize=5)
        pre_img = cv2.medianBlur(pre_img, 5)
        return pre_img

    def grid(self, cir, cap_type=(5, 10)):
        """
        获取网格点...
        cir: 感兴趣圆心      
        cap_type: 瓶子规格
        return: 
        grid: 网格点     
        crop_size: 截取尺寸
        """
        from sklearn.cluster import KMeans
        from itertools import product
        X = cir[:, 1].reshape(-1, 1)
        Y = cir[:, 0].reshape(-1, 1)
        k_means_row = KMeans(
            init='k-means++', n_clusters=cap_type[1], n_init=12)
        k_means_col = KMeans(
            init='k-means++', n_clusters=cap_type[0], n_init=12)
        k_means_row.fit(X)
        k_means_col.fit(Y)
        row = sorted(k_means_row.cluster_centers_[:, 0])
        col = sorted(k_means_col.cluster_centers_[:, 0])

        grid = list(product(col, row))
        grid = np.uint16(np.round(grid))

        crop_row = int((row[-1]-row[0])*0.5/(len(row)-1))
        crop_col = int((col[-1]-col[0])*0.5/(len(col)-1))
        return grid, (crop_col, crop_row)

    def get_canimgs(self, img, cir, crop_size=(45, 45)):
        """
        截取候选区域
        """
        can_imgs = []
        can_marks = []
        h, w, _ = img.shape
        for i in cir:
            can_img = img[max(0, i[1]-crop_size[1]):min(i[1]+crop_size[1], h),
                          max(0, i[0]-crop_size[0]):min(i[0]+crop_size[0], w)]
            can_img = cv2.resize(can_img, (64, 64))
            can_imgs.append(can_img)
            can_marks.append((i[0], i[1]))
        can_imgs = np.array(can_imgs)
        can_marks = np.array(can_marks, dtype=int)
        return can_imgs, can_marks

    def draw_cir(self, img, cir, rad=30, color=(0, 255, 0)):
        """画圆..."""
        for i in cir:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), rad, color, 2)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, color, 3)
        return img

    def draw_rec(self, img, cir, crop_size=(45, 45)):
        h, w = img.shape
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for i in cir:
            cv2.rectangle(color_img, (max(0, i[0]-int(0.8*crop_size[0])), max(0, i[1]-int(0.8*crop_size[1]))),
                          (min(i[0]+int(0.8*crop_size[0]), w), min(i[1]+int(0.8*crop_size[1]), h)), (0, 255, 0), 2)
        return color_img
