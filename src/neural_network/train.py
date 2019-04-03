import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from time 
import random
import glob 
import ops 
import sys

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from batch_loader import BatchLoader


def train_CNN_classifier():
    """
    """
    dg = DataGenerator()
    X_data, Y_data, X_test, Y_test, = dg.train_test_split()
    

    with tf.Graph().as_default(), tf.device('/cpu:0'):


        


if __name__ == '__main__':






    with tf.Graph().as_default(), tf.device('/cpu:0'):

        input_ = tf.placeholder(tf.float32, shape=[None, 44, 534, 1])
        label_ = tf.placeholder(tf.float32, shape=[None, , patch_size, 1])
        train_bl = batch_utils.TrainBatchLoader(train_images, input_, label_, train_config)
        valid_bl = batch_utils.ValidBatchLoader(valid_images, input_, label_, valid_config)

        train_x, train_y = train_bl.get_batch()
        valid_x, valid_y = valid_bl.get_batch()

        with tf.device('/cpu:0'):

            with tf.variable_scope('Network'):
                G = network.Network(train_x, train_config)
            # with tf.variable_scope('Discriminator'):
                # D_fake = network.Discriminator(G.output, train_config)
            # with tf.variable_scope('Discriminator', reuse=True):
                # D_real = network.Discriminator(train_y, train_config)

            # D_fake_loss = tf.reduce_mean(tf.square(D_fake.output))
            # D_real_loss = tf.reduce_mean(tf.square(1.0 - D_real.output))
            # D_loss = D_fake_loss + D_real_loss

            G_SSIM_loss = (1.0-tf.reduce_mean(tf.image.ssim(normalize(train_y), normalize(G.output), max_val = 1.0)))/2.0
            G_MSE_loss = tf.reduce_mean(tf.abs(G.output - train_y))
            G_TV = tf.reduce_mean(tf.image.total_variation(G.output))/ (2 * (train_config.image_size ** 2))

            G_dis_loss = tf.reduce_mean(tf.square(1.0 - D_fake.output))
            G_loss = G_MSE_loss + 0.01 * G_TV

            with tf.variable_scope('Network', reuse=True):
                valid_G = network.Network(valid_x, valid_config)
            # with tf.variable_scope('Discriminator', reuse=True):
                # valid_D_fake = network.Discriminator(valid_G.output, valid_config)
                # valid_D_real = network.Discriminator(valid_y, valid_config)

            # valid_D_fake_loss = tf.reduce_mean(tf.square(valid_D_fake.output))
            # valid_D_real_loss = tf.reduce_mean(tf.square(1 - valid_D_real.output))

            valid_G_SSIM_loss = (1.0-tf.reduce_mean(tf.image.ssim(normalize(valid_y), normalize(valid_G.output), max_val = 1.0)))/2.0
            valid_G_MSE_loss = tf.reduce_mean(tf.abs(valid_G.output - valid_y))

            valid_G_TV = tf.reduce_mean(tf.image.total_variation(valid_G.output ))/ (2 * (valid_config.image_size ** 2))

            valid_G_dis_loss = tf.reduce_mean(tf.square(1.0 - valid_D_fake.output))
            valid_G_loss = valid_G_MSE_loss + 0.01 * valid_G_TV

            gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Network')
            # dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Discriminator')

            G_train_step = tf.train.AdamOptimizer(1e-4).minimize(G_loss, var_list=gen_var_list)
            D_train_step = tf.train.AdamOptimizer(1e-5).minimize(D_loss, var_list=dis_var_list)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=0)
            # saver.restore(sess, 'Models/best_model')

            tf.train.start_queue_runners(sess=sess)
            train_bl.start_threads(sess, n_threads=train_config.n_threads)
            valid_bl.start_threads(sess, n_threads=valid_config.n_threads)
            for i in tqdm(range(25)): sleep(1)
            print(train_bl.queue.size().eval(), valid_bl.queue.size().eval())

            valid_log = open('valid_log.txt', 'w')

            n_eval_steps = valid_config.q_limit // valid_config.batch_size
            check = train_config.checkpoint
            min_loss, min_G_dis_loss = float('inf'), float('inf')
            start_time = time()
            GenNum = 4
            DisNum = 1
            for x in range(1, 1000):
                d_fake_loss, d_real_loss, g_loss = 0, 0, 0
                for i in range(check):
                    for _ in range(GenNum):
                        _, b = sess.run([G_train_step, G_loss])
                        g_loss += b
                    # for _ in range(DisNum):
                        # _, a1, a2 = sess.run([D_train_step, D_fake_loss, D_real_loss])
                        # d_fake_loss += a1
                        # d_real_loss += a2
                    sys.stdout.flush()
                res = np.mean([sess.run([valid_G_loss, valid_G_MSE_loss, valid_G_TV]) for _ in range(n_eval_steps)], axis=0)
                format_str = ('iter: %d valid_G_loss: %.3f valid_G_MSE_loss: %.3f  valid_G_TV: %.3f train_G_loss: %.3f : %.3f time: %d')
                text = (format_str % (x*check, res[0], res[1], res[2], g_loss/(GenNum*check)*100, int(time()-start_time)))
                ops.print_out(valid_log, text)
                saver.save(sess, 'Models_lower/{}'.format(x*check))
                if res[0] < min_loss:
                    min_loss = res[0]
                    saver.save(sess, 'Models_lower/best_model')
                # if d_fake_loss/(DisNum*check)< 0.255 and d_fake_loss/(DisNum*check)> 0.235 and d_real_loss/(DisNum*check)< 0.255 and d_real_loss/(DisNum*check)> 0.235:
                #     DisNum = min(3, DisNum+1)
                # else:
                #     DisNum = max(1, DisNum-1)
