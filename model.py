import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorlayer as tl
import numpy as np
from vgg import *
import time, pdb, os, re

class Model(object):
    def __init__(self, batch_size, width, height, depth, num_classes,
                 total_classes, epoch, print_freq_step, logger, checkpoint_dir = 'checkpoint', net='vgg_16',
                 learning_rate=0.01, beta1=0.9, initializer=None, name_scope='vgg_16'):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.depth = depth
        self.num_classes = num_classes
        self.total_classes = total_classes
        self.epoch = epoch
        self.print_freq_step = print_freq_step
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.net = net
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.name_scope=name_scope
        assert total_classes == np.sum(np.asarray(num_classes, dtype=np.int32))
        self.inputs = tf.placeholder('float32', [batch_size, height, width, depth], name='inputs')
        self.labels = tf.placeholder('int32', [batch_size, total_classes], name='labels')
        self.is_training =tf.placeholder('bool', [], name='is_training')
        # with tf.Graph().as_default():
        label_1, label_2, label_3 = tf.split(self.labels, num_classes, axis=1)
        output, _ = globals()[self.net](self.inputs, self.total_classes, is_training=self.is_training, scope=self.name_scope)
        outputs = tf.split(output, num_classes, axis=3)
        nets = [0] * 3
        with tf.variable_scope(self.name_scope, default_name='vgg_16', initializer=tf.orthogonal_initializer):
            for idx in range(3):
                net = slim.flatten(outputs[idx], scope='flatten_{}'.format(idx))
                nets[idx] = slim.stack(net, slim.fully_connected, [200, 100, num_classes[idx]], scope='fc_{}'.format(idx))
        loss_1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(label_1, nets[0]))
        loss_2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(label_2, nets[1]))
        loss_3 = tf.reduce_mean(tf.losses.softmax_cross_entropy(label_3, nets[2]))
        self.total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('loss', self.total_loss)
        output_1 = tf.argmax(tf.nn.softmax(nets[0]), axis=1)
        output_2 = tf.argmax(tf.nn.softmax(nets[1]), axis=1)
        output_3 = tf.argmax(tf.nn.softmax(nets[2]), axis=1)
        self.nets = nets
        self.output_1 = output_1
        self.output_2 = output_2
        self.output_3 = output_3

        label_1 = tf.argmax(label_1, axis=1)
        label_2 = tf.argmax(label_2, axis=1)
        label_3 = tf.argmax(label_3, axis=1)
        self.prec_1 = tf.metrics.accuracy(label_1, output_1, name='accuracy_1')
        self.prec_2 = tf.metrics.accuracy(label_2, output_2, name='accuracy_2')
        self.prec_3 = tf.metrics.accuracy(label_3, output_3, name='accuracy_3')
        self.recall_1 = tf.metrics.recall(label_1, output_1, name='recall_1')
        self.recall_2 = tf.metrics.recall(label_2, output_2, name='recall_1')
        self.recall_3 = tf.metrics.recall(label_3, output_3, name='recall_1')

        t_vars = tf.trainable_variables()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.total_loss, var_list=t_vars)
        self.saver = tf.train.Saver()

    def train(self,sess, inputs, labels, global_step=0):
        assert self.total_classes == labels.shape[1]
        model_name = self.name_scope + '.model'
        n_batch = 0
        epoch_time = time.time()
        total_loss, total_prec_1, total_prec_2, total_prec_3, total_recall_1, total_recall_2, total_recall_3= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        t_batch = inputs.shape[0] // self.batch_size + 1
        for batch in tl.iterate.minibatches(inputs=inputs, targets=labels,
                                            batch_size=self.batch_size, shuffle=True):
            images, labels = batch
            step_time = time.time()
            _, loss, prec_1, prec_2, prec_3, recall_1, recall_2, recall_3 = sess.run([self.optimizer, self.total_loss,
                                                                                      self.prec_1, self.prec_2, self.prec_3],
                                                                                      feed_dict={self.inputs:images, self.labels:labels,
                                                                                                 self.is_training:True})
            total_loss += loss
            total_prec_1 += prec_1[1]; total_prec_2 += prec_2[1]; total_prec_3 += prec_3[1];
            # total_recall_1 += recall_1[1]; total_recall_2 += recall_2[1]; total_recall_3 += recall_3[1]
            n_batch += 1

            if n_batch % self.print_freq_step == 0:
                self.logger.info("step [%d/%d] loss %f prec_1 %f prec_2 %f prec_3 %f took %fs "
                % (n_batch, t_batch, loss, prec_1[1], prec_2[1], prec_3[1], time.time()-step_time))
                # self.logger.info("step [%d/%d] loss %f prec_1 %f prec_2 %f prec_3 %f recall_1 %f recall_2 %f recall_3 %f took %fs "
                # % (n_batch, t_batch, loss, prec_1[1], prec_2[1], prec_3[1], recall_1[1], recall_2[1], recall_3[1], time.time()-step_time))
                output, output_1, output_2, output_3 = sess.run([self.output, self.output_1, self.output_2, self.output_3], feed_dict={self.inputs:images, self.labels:labels,self.is_training:True})
            if np.isnan(loss):
                exit(" ** NaN loss found during training, stop training")
            # if np.isnan(out).any():
            #     exit(" ** NaN found in output images during training, stop training")
        self.logger.info("**train avg loss %f prec_1 %f prec_2 %f prec_3 %f took %fs global_step %d"
        % (total_loss/n_batch, total_prec_1/n_batch, total_prec_2/n_batch, total_prec_3/n_batch, time.time()-epoch_time, global_step))

        self.saver.save(sess,
                os.path.join(self.checkpoint_dir, model_name),
                global_step=global_step)

    def evaluation(self,sess, inputs, labels, global_step=0):
        assert self.total_classes == labels.shape[1]
        n_batch = 0
        epoch_time = time.time()
        total_loss, total_prec_1, total_prec_2, total_prec_3, total_recall_1, total_recall_2, total_recall_3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for batch in tl.iterate.minibatches(inputs=inputs, targets=labels,
                                            batch_size=self.batch_size, shuffle=True):
            images, labels = batch
            step_time = time.time()
            loss, prec_1, prec_2, prec_3, recall_1, recall_2, recall_3 = sess.run([self.total_loss,
                                                                                   self.prec_1, self.prec_2, self.prec_3,
                                                                                   self.recall_1, self.recall_2, self.recall_3],
                                                                                   feed_dict={self.inputs:images, self.labels:labels,
                                                                                              self.is_training:False})
            total_loss += loss
            total_prec_1 += prec_1[1]; total_prec_2 += prec_2[1]; total_prec_3 += prec_3[1]
            total_recall_1 += recall_1[1]; total_recall_2 += recall_2[1]; total_recall_3 += recall_3[1]
            n_batch += 1

            # if n_batch % self.print_freq_step == 0:
            #     logger.info("step %d loss %d prec_1 %d prec_2 %d prec_3 %d recall_1 %d recall_2 %d recall_3 %d took %fs "
            #     % (n_batch, loss, prec_1, prec_2, prec_3, recall_1, recall_2, recall_3, time.time()-step_time))
            if np.isnan(loss):
                exit(" ** NaN loss found during training, stop training")
            # if np.isnan(out).any():
            #     exit(" ** NaN found in output images during training, stop training")
        self.logger.info("**evaluation avg loss %f prec_1 %f prec_2 %f prec_3 %f took %fs gloabal_step %d"
        % (total_loss/n_batch, total_prec_1/n_batch, total_prec_2/n_batch, total_prec_3/n_batch, time.time()-epoch_time, global_step))

    def load(self):
        print(" [*] Reading checkpoints...")
        model_name = self.name_scope + '.model'
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
          self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
          counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
          print(" [*] Success to read {}".format(ckpt_name))
          return True, counter
        else:
          print(" [*] Failed to find a checkpoint")
          return False, 0
