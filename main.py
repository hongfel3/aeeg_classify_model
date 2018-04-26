import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorlayer as tl
import numpy as np
import os, time, logging, pdb
from model import Model

def main():
    checkpoint_dir = "./checkpoint"
    pretrain_path = "./checkpoint/vgg_16.ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("./log/model.log")
    formatter = logging.Formatter("%(levelname)s-%(message)s")
    fh.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(console)
    # tl.files.exists_or_mkdir("samples/{}".format(task))
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    run_config.gpu_options.allow_growth=True
    data_dir = './data/dataset.npz'
    npzfile = np.load(data_dir)
    x_train, x_test, y_train, y_test = npzfile['x_train'], npzfile['x_test'], npzfile['y_train'], npzfile['y_test']
    assert x_train.shape[0] == y_train.shape[0]

    batch_size = 10
    lr = 0.0001
    w, h = 1500, 300
    depth = 3
    label_size = [3,3,2]
    n_epoch = 100
    print_freq_step = 5
    total_classes = 8
    counter = 0
    pretrain = True
    classification_model = Model(batch_size, w, h, depth, label_size, total_classes, n_epoch, print_freq_step,
                                 logger, checkpoint_dir,net='vgg_16', learning_rate=lr, name_scope='vgg_16')
    sess = tf.Session(config=run_config)
    tf.global_variables_initializer().run(session=sess)
    tf.local_variables_initializer().run(session=sess)
    if pretrain:
        variables_to_restore = slim.get_variables_to_restore()
        slim.assign_from_checkpoint_fn(pretrain_path, variables_to_restore,ignore_missing_vars=True)
        # classification_model.pretrain(model_dir, variables_to_restore, 'vgg_16')
    # could_load, checkpoint_counter = classification_model.load(checkpoint_dir)
    # if could_load:
    #   counter = checkpoint_counter
    #   logger.info(" [*] Load checkpoint dir SUCCESS")
    # else:
    #   print(" [!] Load checkpoint dir failed...")# try:
    #     tf.global_variables_initializer().run(session=sess)
    # except:
    #     tf.initialize_all_variables().run(session=sess)

    for idx in range(1, int(n_epoch)+1):
        logger.info('**epoch [%d/%d]' % (idx, n_epoch))
        classification_model.train(sess, x_train, y_train, global_step=counter+idx)
        classification_model.evaluation(sess, x_test, y_test, global_step=counter+idx)

if __name__ == '__main__':
    main()
