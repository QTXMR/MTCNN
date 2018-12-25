# coding:utf-8

# from train_models.mtcnn_model import P_Net
# from train_models.train import train
from train_models import mtcnn_model
from train_models import train

def train_PNet(data_path, model_path, end_epoch, display, lr):
    """
    train PNet
    :param data_path: tfrecord path
    :param model_path: 模型路径
    :param end_epoch: max epoch for training
    :param display:
    :param lr: learning rate
    :return:
    """
    net_factory = mtcnn_model.P_Net  # TODO:调用mtcnn_model模块中P_Net函数， 返回cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
    train.train(net_factory, model_path, end_epoch, data_path, display=display, base_lr=lr)


if __name__ == '__main__':
    # data path
    data_path = '../data/synthetic_data/PNet'
    model_name = 'MTCNN'
    model_path = '../data/%s_model/PNet' % model_name

    # prefix = model_path
    end_epoch = 30
    display = 100
    lr = 0.001
    train_PNet(data_path, model_path, end_epoch, display, lr)
