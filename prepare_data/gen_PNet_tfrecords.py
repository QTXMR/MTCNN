# coding:utf-8
import os
import random
import sys
# import time
import tensorflow as tf
from prepare_data.tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple

"""
将上合成的数据，生成tfrecord格式的，便于tensorflow训练
"""


def get_dataset(dir, net='PNet'):
    """

    :param dir:
    :param net:
    :return: file name,label and offset
    """
    item = 'synthetic_data/PNet/train_%s.txt' % net
    dataset_dir = os.path.join(dir, item)
    imagelist = open(dataset_dir, 'r')
    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        # get offset info
        bbox['xmin'] = float(info[2])
        bbox['ymin'] = float(info[3])
        bbox['xmax'] = float(info[4])
        bbox['ymax'] = float(info[5])
        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      filename: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def run(dataset_dir, net, output_dir, name='MTCNN', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    # tfrecord name
    tf_filename = output_dir + "/train_" + net + ".tfrecord"
    if tf.gfile.Exists(tf_filename):
        print("Dataset files already exist. Exiting without re-creating them.")
        return
    # get file name , label and offset, and shuffling.
    dataset = get_dataset(dataset_dir, net=net)
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        random.shuffle(dataset)
    # write the data to tfrecord
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i + 1) % 100 == 0:
                sys.stdout.write('\r>> %d/%d images has been converted' % (i + 1, len(dataset)))  # 输出到屏幕上
            sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')


if __name__ == '__main__':
    dir = '../data/'
    net = 'PNet'
    output_directory = '../data/synthetic_data/PNet'
    run(dir, net, output_directory, shuffling=True)
