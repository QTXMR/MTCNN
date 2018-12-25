# coding:utf-8
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from prepare_data.read_tfrecord_v2 import read_multi_tfrecords, read_single_tfrecord
import random
import cv2
from train_models.MTCNN_config import config


# sys.path.append("../prepare_data")
# print(sys.path)


def train_model(base_lr, loss, data_num):
    """
    train model
    :param base_lr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op
    """
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    # LR_EPOCH [8,14]
    # boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    # lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
    # control learning rate
    """
    Piecewise constant from boundaries and interval values.
    Example: use a learning rate that's 1.0 for the first 100001 steps,
     0.5 for the next 10000 steps, and 0.1 for any additional steps.
    """
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    # 优化器
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)  # 0.9在传递给优化器的变量的值上计算梯度。利用Nesterov动量使变量跟踪论文中“theta_t + mu*v_t”的值
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op


# all mini-batch mirror 随机翻转图片
def random_flip_images(image_batch, label_batch):
    # mirror
    if random.choice([0, 1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch == -2)[0]  # TODO:delete landmark
        flipposindexes = np.where(label_batch == 1)[0]
        # only flip
        flipindexes = np.concatenate((fliplandmarkindexes, flipposindexes))
        # random flip
        for i in flipindexes:
            cv2.flip(image_batch[i], 1, image_batch[i])

            # pay attention: flip landmark
        # for i in fliplandmarkindexes:
        #     landmark_ = landmark_batch[i].reshape((-1, 2))
        #     landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
        #     landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
        #     landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth
        #     landmark_batch[i] = landmark_.ravel()

    return image_batch
    # return image_batch, landmark_batch


def image_color_distort(inputs):
    """
    image 亮度，对比度，色调，饱和度随机调整
    :param inputs:
    :return:
    """
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)  # 用随机因素调整图像的对比度
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)  # 用随机因素调整图像的亮度
    inputs = tf.image.random_hue(inputs, max_delta=0.2)  # 用随机因素调整RGB图像的色调
    inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)  # 用随机因素调整RGB图像的饱和度

    return inputs


def train(net_factory, model_path, end_epoch, data_path, display=200, base_lr=0.01):
    """
    train PNet/RNet/ONet
    :param net_factory: 网络模型
    :param model_path: model path
    :param end_epoch:
    :param data_path:data_path
    :param display:
    :param base_lr:
    :return:
    """
    net = model_path.split('/')[-1]
    # label file
    label_file = os.path.join(data_path, 'train_%s.txt' % net)
    f = open(label_file, 'r')
    # get number of training examples
    num = len(f.readlines())
    # print("Total size of the dataset is: ", num)
    # print(model_path)

    # PNet use this method to get data
    if net == 'PNet':
        dataset_dir = os.path.join(data_path, 'train_%s.tfrecord_shuffle' % net)
        image_batch, label_batch, bbox_batch = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, net)

    # RNet use 3 tfrecords to get data
    else:
        pos_dir = os.path.join(data_path, 'pos.tfrecord_shuffle')
        part_dir = os.path.join(data_path, 'part.tfrecord_shuffle')
        neg_dir = os.path.join(data_path, 'neg.tfrecord_shuffle')
        dataset_dirs = [pos_dir, part_dir, neg_dir]
        pos_radio = 1.0 / 6;
        part_radio = 1.0 / 6;
        neg_radio = 3.0 / 6
        pos_batch_size = int(np.ceil(config.BATCH_SIZE * pos_radio))
        assert pos_batch_size != 0, "Batch Size Error "
        part_batch_size = int(np.ceil(config.BATCH_SIZE * part_radio))
        assert part_batch_size != 0, "Batch Size Error "
        neg_batch_size = int(np.ceil(config.BATCH_SIZE * neg_radio))
        assert neg_batch_size != 0, "Batch Size Error "
        batch_sizes = [pos_batch_size, part_batch_size, neg_batch_size]
        # print('batch_size is:', batch_sizes)
        image_batch, label_batch, bbox_batch = read_multi_tfrecords(dataset_dirs, batch_sizes, net)

    if net == 'PNet':
        image_size = 12
        radio_cls_loss = 1.0;  # importance 这表示任务的重要程度
        radio_bbox_loss = 0.5;
    elif net == 'RNet':
        image_size = 24
        radio_cls_loss = 1.0;
        radio_bbox_loss = 0.5;
    else:
        radio_cls_loss = 1.0;
        radio_bbox_loss = 0.5;
        image_size = 48

    # define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')

    # image 亮度，对比度，色调，饱和度随机调整
    input_image = image_color_distort(input_image)
    # TODO: 调用Net，传入数据，得到loss_op
    # get loss and accuracy
    cls_loss_op, bbox_loss_op, L2_loss_op, accuracy_op = net_factory(input_image, label, bbox_target, training=True)
    # train,update learning rate(3 loss)
    total_loss_op = radio_cls_loss * cls_loss_op + radio_bbox_loss * bbox_loss_op + L2_loss_op
    # TODO:训练模型
    train_op, lr_op = train_model(base_lr, total_loss_op, num)
    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()

    # save model
    saver = tf.train.Saver(max_to_keep=0)  # checkpoint有何用处
    sess.run(init)

    # visualize some variables 可视化输出
    tf.summary.scalar("cls_loss", cls_loss_op)  # cls_loss
    tf.summary.scalar("bbox_loss", bbox_loss_op)  # bbox_loss
    tf.summary.scalar("cls_accuracy", accuracy_op)  # cls_acc
    tf.summary.scalar("total_loss", total_loss_op)  # cls_loss, bbox loss, landmark loss and L2 loss add together
    summary_op = tf.summary.merge_all()
    logs_dir = "../logs/%s" % (net)
    if os.path.exists(logs_dir) == False:
        os.mkdir(logs_dir)
    writer = tf.summary.FileWriter(logs_dir, sess.graph)
    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer, projector_config)
    # begin 创建一个新的协调器。
    coord = tf.train.Coordinator()
    # begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    # total steps
    MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
    epoch = 0
    sess.graph.finalize()
    try:
        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array = sess.run(
                [image_batch, label_batch, bbox_batch])
            # random flip
            image_batch_array = random_flip_images(image_batch_array, label_batch_array)
            _, _, summary = sess.run([train_op, lr_op, summary_op],
                                     feed_dict={input_image: image_batch_array, label: label_batch_array,
                                                bbox_target: bbox_batch_array})

            if (step + 1) % display == 0:
                # acc = accuracy(cls_pred, labels_batch)
                cls_loss, bbox_loss, L2_loss, lr, acc = sess.run(
                    [cls_loss_op, bbox_loss_op, L2_loss_op, lr_op, accuracy_op],
                    feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})

                total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + L2_loss
                # landmark loss: %4f,
                print("%s : Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                        datetime.now(), step + 1, MAX_STEP, acc, cls_loss, bbox_loss, L2_loss, total_loss, lr))

            # save every two epochs TODO: 保存模型，到相应的目录
            if i * config.BATCH_SIZE > num * 2:
                epoch = epoch + 1
                i = 0
                #TODO:保存模型
                path_prefix = saver.save(sess, model_path, global_step=epoch * 2)
                print('path prefix is :', path_prefix)
            writer.add_summary(summary, global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()
