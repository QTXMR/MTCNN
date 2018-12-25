#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
labelme工具生成json文件解析
"""
import os
import json
import jsonpath


def list_file(data_dir, suffix):
    """
    选出json文件
    :param data_dir: 文件路径
    :param suffix: json文件后缀
    :return: 后缀为.json的文件名列表
    """
    fs = os.listdir(data_dir) # 返回指定路径下的文件和文件夹列表。
    for i in range(len(fs) - 1, -1, -1):
        # 如果后缀不是.json就将该文件删除掉
        if not fs[i].endswith(suffix):
            del fs[i]
    return fs


def write_label(data_dir, label):
    """
    将json文件中的点位置信息检索出，写入label.txt文件中
    :param data_dir: json文件所在路径
    """
    jsons = list_file(data_dir, ".json")
    for file_name in jsons:
        file_name = os.path.join(data_dir + "\\" + file_name)  # 拿到json文件
        jsonfile = json.load(open(file_name))  # 打开json文件
        point_list = jsonpath.jsonpath(jsonfile, "$..points")  # 读取点的信息
        # print(jsonfile["shapes"][i]["points"])
        image_path = data_dir + "/" + jsonfile["imagePath"]  # 读取图片名合成图片所在路径
        line = image_path
        # count = 0  # 判断框的个数是否一致，后期需要屏蔽
        for points in point_list:
            # count += 1
            for point in points:
                for i in point:
                    line = line + " " + str(i)

        # line = str(count) + " " + line + "\n"
        line = line + "\n"
        label.write(line)


def resolve_json(data_dir, label_dir):
    """
    封装调用接口
    :param data_dir: json文件路径
    :param label_dir: label.txt路径
    """
    label_file = open(label_dir, "w")
    write_label(data_dir, label_file)
    label_file.close()


if __name__ == '__main__':
    print("解析开始，请稍等……")
    data_dir = "../data/RM-RGB"  # 处理文件所在路径
    label_dir = "../data/label_train.txt"
    resolve_json(data_dir, label_dir)
    print("解析完成")
