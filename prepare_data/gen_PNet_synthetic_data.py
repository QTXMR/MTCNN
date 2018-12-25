import numpy as np
import numpy.random as npr
import os

"""
获取上一步生成的数据，随机合成中间数据
"""


def open_file(size):
    """
    创建或打开所需要的文件
    :param size: 图片的尺寸，用来识别网络PNet,RNet,ONet
    :return: dir_path, net, neg, pos, part
    """
    data_dir = '../data'
    if size == 12:
        net = "PNet"
    elif size == 24:
        net = "RNet"
    elif size == 48:
        net = "ONet"

    with open(os.path.join(data_dir, '%s/pos_%s.txt' % (size, size)), 'r') as f:
        pos = f.readlines()

    with open(os.path.join(data_dir, '%s/neg_%s.txt' % (size, size)), 'r') as f:
        neg = f.readlines()

    with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:
        part = f.readlines()

    dir_path = os.path.join(data_dir, 'synthetic_data')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if not os.path.exists(os.path.join(dir_path, "%s" % (net))):
        os.makedirs(os.path.join(dir_path, "%s" % (net)))

    return dir_path, net, neg, pos, part


def gen_imglist(size):
    """

    :param size:
    """
    dir_path, net, neg, pos, part = open_file(size)
    with open(os.path.join(dir_path, "%s" % (net), "train_%s.txt" % (net)), "w") as f:
        nums = [len(neg), len(pos), len(part)]
        ratio = [3, 1, 1]
        # base_num = min(nums)
        # TODO:
        base_num = 25000
        # print(len(neg), len(pos), len(part), base_num)

        # shuffle the order of the initial data
        # if negative examples are more than 750k then only choose 750k
        if len(neg) > base_num * 3:
            # neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)  # 返回数据中随机选择的一个数
            neg_keep = npr.choice(len(neg), size=len(neg), replace=True)  # 返回数据中随机选择的一个数
        else:
            neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
            # neg_keep = npr.choice(len(neg), size=base_num, replace=True)
        pos_keep = npr.choice(len(pos), size=base_num, replace=True)
        part_keep = npr.choice(len(part), size=base_num, replace=True)
        # print(len(neg_keep), len(pos_keep), len(part_keep))

        # # write the data according to the shuffled order
        # for i in pos_keep:
        #     f.write(pos[i])
        # # 因为negative样本没有偏移位置信息，需要保持统一的格式
        # for j in neg_keep:
        #     p = neg[j].find(" ") + 1
        #     neg[j] = neg[j][:p - 1] + " " + neg[j][p:-1] + " -1 -1 -1 -1\n"
        #     f.write(neg[j])
        # for k in part_keep:
        #     f.write(part[k])

        # # 不是用乱序的数据
        for i in range(int(len(pos))):
            p = pos[i].find(" ") + 1
            pos[i] = pos[i][:p - 1] + " " + pos[i][p:-1] + "\n"
            f.write(pos[i])

        for i in range(int(len(neg))):
            p = neg[i].find(" ") + 1
            neg[i] = neg[i][:p - 1] + " " + neg[i][p:-1] + " -1 -1 -1 -1\n"
            f.write(neg[i])

        for i in range(int(len(part))):
            p = part[i].find(" ") + 1
            part[i] = part[i][:p - 1] + " " + part[i][p:-1] + "\n"
            f.write(part[i])


if __name__ == '__main__':
    size = 12
    gen_imglist(size)
    print("数据合并完成，请用其它软件查看数据，pycharm加载不完整")
