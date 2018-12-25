# coding:utf-8
import sys
import os
import cv2
import numpy as np
import numpy.random as npr
from prepare_data.utils import IoU

""" 
此模块是使用标签数据和原始图片，生成negative，part,positive 等训练数据
"""


def generate_path(dir):
    """
    检查和创建目录
    :param dir: 创建的目录
    """
    if not os.path.exists(dir):  # exists()路径存在则返回True,路径损坏返回False
        os.mkdir(dir)  # 创建一个名为save_dir的文件夹.


def open_file(dir, file_name):
    """
    打开路径下的文件，没有的自动创建
    :param dir: 路径
    :param file_name: 文件名
    :return: 文件引用
    """
    file = open(os.path.join(dir, file_name), "w")  # os.path.join 把目录和文件名合成一个路径
    return file


def load_image(im_path, image_idx):
    """
    load image
    :param im_path: 图片路径
    :return: 图片
    """
    img = cv2.imread(im_path)
    if img is None:
        print("No image")
        sys.exit(1)
    image_idx += 1
    return img, image_idx


def generate_negative_sample(height, width, boxes, img, nega_idx, stdsize=12):
    # 1---->50
    # keep crop random parts, until have 50 negative examples
    # get 50 negative sample from every image
    neg_num = 0
    while neg_num < 100:
        # TODO: 酌情修改 neg_num's size [12,min(width, height) / 2],min_size:12
        # size is a random number between 12 and min(width,height)
        size = npr.randint(12, min(width, height) / 2)
        # top_left coordinate
        nx = npr.randint(0, width - size)  # 产生 0到 width - 12 的一个整数型随机数
        ny = npr.randint(0, height - size)  # 产生 0到 height - 12 的一个整数型随机数
        # random crop
        crop_box = np.array([nx, ny, nx + size, ny + size])
        # calculate iou
        Iou = IoU(crop_box, boxes)

        # crop a part from inital image
        cropped_im = img[ny: ny + size, nx: nx + size, :]
        # resize the cropped image to size 12*12
        resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)  # 插值=双线性插值,调整图像大小

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % nega_idx)
            # nega_file.write("../data/12/negative/%s.jpg" % nega_idx + ' 0\n')
            # nega_file.write(save_file + ' 0\n')  # 不用这语句，是因为会自动产生 "../data/12/negative\%s.jpg"的目录
            # but 可以这样
            file_path, file_name = os.path.split(save_file)
            nega_file.write(file_path + "/%s.jpg" % nega_idx + " 0\n")
            cv2.imwrite(save_file, resized_im)
            nega_idx += 1
            neg_num += 1
    return nega_idx


def gen_negative_sample(height, width, x1, y1, w, h, boxes, img, nega_idx, stdsize=12):
    for i in range(5):
        # size of the image to be cropped
        size = npr.randint(12, min(width, height) / 2)
        # delta_x and delta_y are offsets of (x1, y1)
        # max can make sure if the delta is a negative number , x1+delta_x >0
        # parameter high of randint make sure there will be intersection between bbox and cropped_box
        delta_x = npr.randint(max(-size, -x1), w)  # width图片宽度，产生 -x1到 x2-x1 的一个整数型随机数
        delta_y = npr.randint(max(-size, -y1), h)  # height图片宽度，产生 -y1到 y2-y1 的一个整数型随机数
        # max here not really necessary
        nx1 = int(max(0, x1 + delta_x))  # 0到x2
        ny1 = int(max(0, y1 + delta_y))  # 0到y2
        # if the right bottom point is out of image then skip
        if nx1 + size > width or ny1 + size > height:
            continue
        crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
        # TODO:resize cropped image to be 12 * 12
        resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % nega_idx)
            file_path, file_name = os.path.split(save_file)
            nega_file.write(file_path + "/%s.jpg" % nega_idx + "0\n")
            # nega_file.write("../data/12/negative/%s.jpg" % nega_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            nega_idx += 1
    return nega_idx


def gen_positive_sample(height, width, x1, y1, x2, y2, w, h, box, img, posi_idx, part_idx, stdsize=12):
    for i in range(20):
        # pos and part face size [minsize*0.8,maxsize*1.25]
        size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))  # np.ceil 返回接近的最大整数

        # delta here is the offset of box center
        if w < 5:
            # print(w)
            continue
        # print (box)
        delta_x = npr.randint(-w * 0.2, w * 0.2)
        delta_y = npr.randint(-h * 0.2, h * 0.2)

        # show this way: nx1 = max(x1+w/2-size/2+delta_x)
        # x1+ w/2 is the central point, then add offset , then deduct size/2
        # deduct size/2 to make sure that the right bottom corner will be out of
        nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
        # show this way: ny1 = max(y1+h/2-size/2+delta_y)
        ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
        nx2 = nx1 + size
        ny2 = ny1 + size

        if nx2 > width or ny2 > height:
            continue
        crop_box = np.array([nx1, ny1, nx2, ny2])
        # yu gt de offset
        offset_x1 = (x1 - nx1) / float(size)
        offset_y1 = (y1 - ny1) / float(size)
        offset_x2 = (x2 - nx2) / float(size)
        offset_y2 = (y2 - ny2) / float(size)
        # crop
        cropped_im = img[ny1: ny2, nx1: nx2, :]
        # resize
        resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

        box_ = box.reshape(1, -1)
        iou = IoU(crop_box, box_)
        if iou >= 0.65:
            save_file = os.path.join(pos_save_dir, "%s.jpg" % posi_idx)
            file_path, file_name = os.path.split(save_file)
            # nega_file.write(file_path + "/%s.jpg" % nega_idx + "0\n")
            posi_file.write(file_path + "/%s.jpg" % posi_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (
                offset_x1, offset_y1, offset_x2, offset_y2))
            cv2.imwrite(save_file, resized_im)
            posi_idx += 1
        elif iou >= 0.4:
            save_file = os.path.join(part_save_dir, "%s.jpg" % part_idx)
            file_path, file_name = os.path.split(save_file)
            part_file.write(file_path + "/%s.jpg" % part_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (
                offset_x1, offset_y1, offset_x2, offset_y2))
            cv2.imwrite(save_file, resized_im)
            part_idx += 1
    return posi_idx, part_idx


def load_annotation_data(stdsize=12):
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)

    image_idx = 0  # 记录图片的数量
    posi_idx = 0  # positive
    nega_idx = 0  # negative
    part_idx = 0  # 记录part
    box_idx = 0  # 记录框数目

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        # image path
        im_path = annotation[0]
        # boxed change to float type
        bbox = list(map(float, annotation[1:]))
        # gt
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)  # 如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
        img, image_idx = load_image(im_path, image_idx)
        height, width, channel = img.shape

        # get 100 negative sample from every image
        nega_idx = generate_negative_sample(height, width, boxes, img, nega_idx, stdsize)

        # for every bounding boxes
        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            # gt's width
            w = x2 - x1 + 1
            # gt's height
            h = y2 - y1 + 1

            # TODO:可以修改值，忽略较小的框，较小的框数据不准确的
            if max(w, h) < 20 or x1 < 0 or y1 < 0:  # max() 返回给定参数中的最大值
                continue

            # crop another 5 images near the bounding box if IoU less than 0.5, save as negative samples
            # TODO:
            # nega_idx = gen_negative_sample(height, width, x1, y1, w, h, boxes, img, nega_idx,stdsize)
            # generate positive examples and part faces
            # TODO:
            posi_idx, part_idx = gen_positive_sample(height, width, x1, y1, x2, y2, w, h, box, img, posi_idx, part_idx,
                                                     stdsize)
            box_idx += 1
    return image_idx, posi_idx, part_idx, nega_idx


if __name__ == '__main__':
    print("开始生成数据，请稍等……")
    resize = 12
    anno_file = "../data/label_train.txt"
    save_dir = "../data/12"
    pos_save_dir = "../data/12/positive"
    part_save_dir = "../data/12/part"
    neg_save_dir = '../data/12/negative'

    generate_path(save_dir)
    generate_path(pos_save_dir)
    generate_path(part_save_dir)
    generate_path(neg_save_dir)

    posi_file = open_file(save_dir, 'pos_12.txt')
    nega_file = open_file(save_dir, 'neg_12.txt')
    part_file = open_file(save_dir, 'part_12.txt')

    image_idx, posi_idx, part_idx, nega_idx = load_annotation_data()

    print("%s images done, pos: %s part: %s neg: %s" % (image_idx, posi_idx, part_idx, nega_idx))
    print("数据创建完成")

    posi_file.close()
    nega_file.close()
    part_file.close()
