from infogan.models.regularized_gan import RegularizedGAN
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from infogan.misc.distributions import Bernoulli, Gaussian, Categorical
import sys

TINY = 1e-8


class InfoGANTrainer(object):
    def __init__(self,
                 model,
                 batch_size,
                 dataset=None,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 max_epoch=100,
                 updates_per_epoch=100,
                 snapshot_interval=2500,
                 info_reg_coeff=1.0,
                 discriminator_learning_rate=2e-4,
                 generator_learning_rate=2e-4,
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.info_reg_coeff = info_reg_coeff
        self.discriminator_trainer = None
        self.generator_trainer = None
        self.input_tensor = None
        self.log_vars = []

    def init_opt(self):
        self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_dim])

        with pt.defaults_scope(phase=pt.Phase.train):
            dim = 10
            tt = Categorical(dim)
            zz = tt.sample_prior(self.batch_size)
            zzz = 1 - zz
            z = [tf.concat(1, [tf.reshape(zz[:, i], (self.batch_size, 1)), tf.reshape(zzz[:, i], (self.batch_size, 1))])
                 for i in range(dim)]
            disc_z_var = tf.reshape(z, (self.batch_size, 20))

            orig_z_var = self.model.latent_dist.sample_prior(self.batch_size)
            z_var = tf.concat(1, [tf.reshape(orig_z_var[:, 0], (self.batch_size, 1)),disc_z_var, orig_z_var[:, -2:]])

            fake_x, _ = self.model.generate(z_var)
            real_d, _, _, _ = self.model.discriminate(input_tensor)
            fake_d, _, fake_reg_z_dist_info, _ = self.model.discriminate(fake_x)

            reg_z = self.model.reg_z(z_var)

            discriminator_loss = - tf.reduce_mean(tf.log(real_d + TINY) + tf.log(1. - fake_d + TINY))
            generator_loss = - tf.reduce_mean(tf.log(fake_d + TINY))

            self.log_vars.append(("discriminator_loss", discriminator_loss))
            self.log_vars.append(("generator_loss", generator_loss))

            mi_est = tf.constant(0.)
            cross_ent = tf.constant(0.)

            # compute for discrete and continuous codes separately
            # discrete:
            if len(self.model.reg_disc_latent_dist.dists) > 0:
                disc_reg_z = self.model.disc_reg_z(reg_z)
                disc_reg_dist_info = self.model.disc_reg_dist_info(fake_reg_z_dist_info)
                disc_log_q_c_given_x = self.model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)
                disc_log_q_c = self.model.reg_disc_latent_dist.logli_prior(disc_reg_z)
                disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
                disc_ent = tf.reduce_mean(-disc_log_q_c)
                disc_mi_est = disc_ent - disc_cross_ent
                mi_est += disc_mi_est
                cross_ent += disc_cross_ent
                self.log_vars.append(("MI_disc", disc_mi_est))
                self.log_vars.append(("CrossEnt_disc", disc_cross_ent))
                discriminator_loss -= self.info_reg_coeff * disc_mi_est
                generator_loss -= self.info_reg_coeff * disc_mi_est

            if len(self.model.reg_cont_latent_dist.dists) > 0:
                cont_reg_z = self.model.cont_reg_z(reg_z)
                cont_reg_dist_info = self.model.cont_reg_dist_info(fake_reg_z_dist_info)
                cont_log_q_c_given_x = self.model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)
                cont_log_q_c = self.model.reg_cont_latent_dist.logli_prior(cont_reg_z)
                cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
                cont_ent = tf.reduce_mean(-cont_log_q_c)
                cont_mi_est = cont_ent - cont_cross_ent
                mi_est += cont_mi_est
                cross_ent += cont_cross_ent
                self.log_vars.append(("MI_cont", cont_mi_est))
                self.log_vars.append(("CrossEnt_cont", cont_cross_ent))
                discriminator_loss -= self.info_reg_coeff * cont_mi_est
                generator_loss -= self.info_reg_coeff * cont_mi_est

            for idx, dist_info in enumerate(self.model.reg_latent_dist.split_dist_info(fake_reg_z_dist_info)):
                if "stddev" in dist_info:
                    self.log_vars.append(("max_std_%d" % idx, tf.reduce_max(dist_info["stddev"])))
                    self.log_vars.append(("min_std_%d" % idx, tf.reduce_min(dist_info["stddev"])))

            self.log_vars.append(("MI", mi_est))
            self.log_vars.append(("CrossEnt", cross_ent))

            all_vars = tf.trainable_variables()
            d_vars = [var for var in all_vars if var.name.startswith('d_')]
            g_vars = [var for var in all_vars if var.name.startswith('g_')]

            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))

            discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
            self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer, losses=[discriminator_loss],
                                                            var_list=d_vars)

            generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
            self.generator_trainer = pt.apply_optimizer(generator_optimizer, losses=[generator_loss], var_list=g_vars)

            for k, v in self.log_vars:
                tf.summary.scalar(k, v)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                self.visualize_all_factors()

    def visualize_all_factors(self):
        with tf.Session():
            for dist_idx in range(4):
                cat = [[0, 1] * i + [1, 0] + [0, 1] * (9 - i) for i in range(10)]
                zz = np.tile(np.asarray(cat), (10, 1))
                cat = [[0, 1] * i + [1, 0] + [0, 1] * (9 - i) for i in range(10)]
                zz = np.tile(np.asarray(cat), (10, 1))
                cont = np.asarray([[i / 10.] * 10 for i in range(10)]).reshape(100, 1)
                if dist_idx==0:
                    z_var = np.concatenate([np.zeros((100, 1)), zz, np.zeros((100, 1)), cont[:]], axis=1)
                elif dist_idx == 1:
                    z_var = np.concatenate([np.zeros((100, 1)), zz, np.ones((100, 1)), cont[:]], axis=1)
                elif dist_idx == 2:
                    z_var = np.concatenate([np.zeros((100, 1)), cont[:], zz, np.zeros((100, 1))], axis=1)
                elif dist_idx == 3:
                    z_var = np.concatenate([np.zeros((100, 1)), cont[:], zz, np.ones((100, 1))], axis=1)

                _, x_dist_info = self.model.generate(z_var.tolist())

                # just take the mean image
                if isinstance(self.model.output_dist, Bernoulli):
                    img_var = x_dist_info["p"]
                elif isinstance(self.model.output_dist, Gaussian):
                    img_var = x_dist_info["mean"]
                else:
                    raise NotImplementedError
                img_var = self.dataset.inverse_transform(img_var)
                rows = 10
                img_var = tf.reshape(img_var, [100] + list(self.dataset.image_shape))
                img_var = img_var[:rows * rows, :, :, :]
                imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
                stacked_img = []
                for row in xrange(rows):
                    row_img = []
                    for col in xrange(rows):
                        row_img.append(imgs[row, col, :, :, :])
                    stacked_img.append(tf.concat(1, row_img))
                imgs = tf.concat(0, stacked_img)
                imgs = tf.expand_dims(imgs, 0)
                tf.summary.image("image_%d_" % (dist_idx), imgs)


    def train(self):

        self.init_opt()

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            saver = tf.train.Saver()

            counter = 0

            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                all_log_vals = []
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    x, _ = self.dataset.train.next_batch(self.batch_size)
                    feed_dict = {self.input_tensor: x}
                    log_vals = sess.run([self.discriminator_trainer] + log_vars, feed_dict)[1:]
                    sess.run(self.generator_trainer, feed_dict)
                    all_log_vals.append(log_vals)
                    counter += 1

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                x, _ = self.dataset.train.next_batch(self.batch_size)

                summary_str = sess.run(summary_op, {self.input_tensor: x})
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_dict = dict(zip(log_keys, avg_log_vals))

                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")


    def generating(self):
        self.init_opt()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # load model
            model_name = '/home/hope-yao/Documents/InfoGAN/ckt/mnist/mnist_2017_02_05_22_41_42/mnist_2017_02_05_22_41_42_5000.ckpt.meta'
            saver = tf.train.Saver()
            new_saver = tf.train.import_meta_graph(model_name)
            saver.restore(sess,
                          '/home/hope-yao/Documents/InfoGAN/ckt/mnist/mnist_2017_02_05_22_41_42/mnist_2017_02_05_22_41_42_5000.ckpt')

            a = np.asarray([[x / 10., y / 10.] for x in range(10) for y in range(10)])
            b = np.tile([1] + [0] * 9, (100, 1))
            c = np.concatenate((np.zeros((100, 62)), b, a), axis=1)
            x_ave, _ = self.model.generate(c.tolist())
            imgs = x_ave.eval().reshape(10, 10, 28, 28)
            import seaborn
            import matplotlib.pyplot as plt
            plt.figure()
            for i in range(10):
                for j in range(10):
                    plt.subplot(10, 10, 10 * i + j + 1)
                    fig = plt.imshow(imgs[i, j, :, :])
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
            plt.show()

    def regen(self):
        self.init_opt()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # load model
            model_name = '/home/hope-yao/Documents/InfoGAN/ckt/mnist/mnist_2017_02_06_15_15_22/mnist_2017_02_06_15_15_22_5000.ckpt.meta'
            saver = tf.train.Saver()
            new_saver = tf.train.import_meta_graph(model_name)
            saver.restore(sess,
                          '/home/hope-yao/Documents/InfoGAN/ckt/mnist/mnist_2017_02_06_15_15_22/mnist_2017_02_06_15_15_22_5000.ckpt')

            input_tensor,input_label = self.dataset.train.next_batch(self.batch_size)

            real_d, _, real_reg_z_dist_info, real_reg_dist_flat = self.model.discriminate(input_tensor)

            ss = np.ones((self.batch_size, 1))

            s1 = real_reg_z_dist_info['id_0_prob']

            mean = real_reg_z_dist_info['id_1_mean']
            stddev = real_reg_z_dist_info['id_1_stddev']
            epsilon = tf.random_normal(tf.shape(mean))
            s0 = mean + epsilon * stddev

            mean = real_reg_z_dist_info['id_2_mean']
            stddev = real_reg_z_dist_info['id_2_stddev']
            epsilon = tf.random_normal(tf.shape(mean))
            s4 = mean + epsilon * stddev

            z_var_reduce = tf.concat(1, [ss, s1, s0, s4])
            #
            x_ave, _ = self.model.generate(z_var_reduce)

            x_in = input_tensor
            x_regen = x_ave.eval()
            z_regen = z_var_reduce.eval()

            import seaborn
            import matplotlib.pyplot as plt

            ii = 12
            # x
            plt.figure()
            plt.imshow(x_in[ii].reshape(28, 28))
            # x->c
            print(z_regen[ii])
            # x->c->x
            plt.figure()
            plt.imshow(x_regen[ii].reshape(28, 28))
            # x->c->x->c
            real_d, _, real_reg_z_dist_info, real_reg_dist_flat = self.model.discriminate(x_regen[ii])
            print('z_regen_regen')
            # print('%0.16f' % real_d.eval())
            # print('%0.16f' % real_reg_z_dist_info['id_0_prob'].eval()[0][0])
            # print('%0.16f' % real_reg_z_dist_info['id_1_prob'].eval()[0][0])
            # if CLASS == 3:
            #     print('%0.16f' % real_reg_z_dist_info['id_2_prob'].eval()[0][0])
            #     print('%0.16f' % real_reg_z_dist_info['id_3_mean'].eval())
            # else:
            #     print('%0.16f' % real_reg_z_dist_info['id_2_mean'].eval())
            print('done')
