from infogan.models.regularized_gan import RegularizedGAN
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from infogan.misc.distributions import Bernoulli, Gaussian, Categorical
import sys

TINY = 2e-8
CLASS = 2

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
                 snapshot_interval=3000,
                 info_reg_coeff=1.0,
                 discriminator_learning_rate=2e-4,
                 generator_learning_rate=2e-4,
                 has_classifier = False,
                 pretrain_classifier = False,
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
        self.has_classifier = has_classifier
        if self.has_classifier:
            self.pretrain_classifier = pretrain_classifier
        else:
            self.pretrain_classifier = False

    def init_opt(self):

        self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_dim])
        if self.has_classifier:
            if self.model.network_type=='rec_crs':
                self.input_label = input_label= tf.placeholder(tf.float32, [self.batch_size, 3]) # 3 different classes
                # self.penalty = input_label[:, 0] + input_label[:, 1] + input_label[:, 2] * 50
                self.penalty = tf.ones((self.batch_size,1))
            elif self.model.network_type == 'rec_crs2':
                self.input_label = input_label = tf.placeholder(tf.float32,[self.batch_size, 2])  # 3 different classes
                self.penalty = tf.ones((self.batch_size,1))
            else:
                self.input_label = input_label= tf.placeholder(tf.float32, [self.batch_size, 10]) # 10 different classes

        with pt.defaults_scope(phase=pt.Phase.train):
            z_var = self.model.latent_dist.sample_prior(self.batch_size)
            if self.model.network_type=='rec_crs':
                s1 = tf.slice(z_var, [0, 0], [self.batch_size, 1])  # noise z
                s2 = tf.slice(z_var, [0, 1], [self.batch_size, 1])
                s3 = tf.slice(z_var, [0, 3], [self.batch_size, 1])
                s4 = tf.slice(z_var, [0, 5], [self.batch_size, 1])
                s5 = tf.slice(z_var, [0, 7], [self.batch_size, 1])  # continuous c
                z_var_reduce = tf.concat(1, [s1, s2, s3, s4, s5])
                fake_x, aaa = self.model.generate(z_var_reduce )
                real_d, _, real_reg_z_dist_info, real_reg_dist_flat = self.model.discriminate(input_tensor)
                fake_d, _, fake_reg_z_dist_info, fake_reg_dist_flat = self.model.discriminate(fake_x)
            elif self.model.network_type == 'rec_crs2':
                s1 = tf.slice(z_var, [0, 0], [self.batch_size, 1])  # noise z
                s2 = tf.slice(z_var, [0, 1], [self.batch_size, 1])
                s3 = tf.slice(z_var, [0, 3], [self.batch_size, 1])
                s4 = tf.slice(z_var, [0, 5], [self.batch_size, 11])
                z_var_reduce = tf.concat(1, [s1, s2, s3, s4])
                fake_x, aaa = self.model.generate(z_var_reduce)
                real_d, _, real_reg_z_dist_info, real_reg_dist_flat = self.model.discriminate(input_tensor)
                fake_d, _, fake_reg_z_dist_info, fake_reg_dist_flat = self.model.discriminate(fake_x)
            else:
                fake_x, _ = self.model.generate(z_var)
                real_d, _, real_reg_z_dist_info, _ = self.model.discriminate(input_tensor)
                fake_d, _, fake_reg_z_dist_info, fake_reg_dist_flat = self.model.discriminate(fake_x)


            discriminator_loss = - tf.reduce_mean(tf.log(real_d + TINY) + tf.log(1. - fake_d + TINY))
            generator_loss = - tf.reduce_mean(tf.log(fake_d + TINY))

            self.log_vars.append(("discriminator_loss", tf.reduce_mean(discriminator_loss )))
            self.log_vars.append(("generator_loss", tf.reduce_mean(generator_loss )))

            if self.has_classifier:
                if self.model.network_type=='rec_crs' or self.model.network_type=='rec_crs2':
                    # tt = self.model.disc_reg_dist_info(real_reg_z_dist_info)
                    tt0 = real_reg_z_dist_info['id_1_prob']
                    tt1 = real_reg_z_dist_info['id_2_prob']
                    if CLASS == 3:
                        tt2 = real_reg_z_dist_info['id_3_prob']
                        # prediction = tf.concat(1, [tt0[:,0], tt1[:,0], tt2[:,0]])
                        prediction = tf.concat(1, [tf.reshape(tt0[:, 0], (self.batch_size, 1)), tf.reshape(tt1[:, 0], (self.batch_size, 1)),
                                                   tf.reshape(tt2[:, 0], (self.batch_size, 1))])
                    else:
                        prediction = tf.concat(1, [tf.reshape(tt0[:, 0], (self.batch_size, 1)),
                                                   tf.reshape(tt1[:, 0], (self.batch_size, 1)),
                                                   ])
                else:
                    prediction = self.model.disc_reg_dist_info(real_reg_z_dist_info)['id_0_prob']
                self.classifier_loss  = classifier_loss  = tf.reduce_sum(input_label * -tf.log(prediction + TINY) + (1 - input_label) * -tf.log(1 - prediction + TINY))/self.batch_size
                discriminator_loss += (classifier_loss*100)
                self.log_vars.append(("classifier_loss", classifier_loss))

            mi_est = tf.constant(0.)
            cross_ent = tf.constant(0.)

            if 1:
                reg_z = self.model.reg_z(z_var)
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
                # continuous:
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
            else:
                mean = real_reg_z_dist_info['id_0_mean']
                stddev = real_reg_z_dist_info['id_0_stddev']
                epsilon = tf.random_normal(tf.shape(mean))
                s0 = mean + epsilon * stddev

                mean = real_reg_z_dist_info['id_3_mean']
                stddev = real_reg_z_dist_info['id_3_stddev']
                epsilon = tf.random_normal(tf.shape(mean))
                s4 = mean + epsilon * stddev

                s1 = real_reg_z_dist_info['id_1_prob']
                s2 = real_reg_z_dist_info['id_2_prob']
                z_var_reduce = tf.concat(1, [tf.reshape(s0[:, 0], (self.batch_size, 1)),
                                             tf.reshape(s1[:, 0], (self.batch_size, 1)),
                                             tf.reshape(s2[:, 0], (self.batch_size, 1)),
                                             s4])
                x_ave, _ = self.model.generate(z_var_reduce)
                self.R1 = R1 = tf.reduce_sum(input_tensor * -tf.log(x_ave+ TINY) + (1 - input_tensor) * -tf.log(1 - x_ave+ TINY))  / self.batch_size / 28 ** 2
                self.log_vars.append(("R1_loss", R1))
                R1_er = tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.pow(input_tensor - x_ave , 2.),1) , self.penalty)) / self.batch_size / 28 ** 2
                self.log_vars.append(("recon_er", R1_er))
                # discriminator_loss += R1*5
                # generator_loss += R1*5

                # KL divergence for VAE-GAN
                sigma0 = real_reg_z_dist_info['id_0_stddev']
                mu0 = real_reg_z_dist_info['id_0_mean']
                sigma1 = real_reg_z_dist_info['id_3_stddev']
                mu1 = real_reg_z_dist_info['id_3_mean']
                self.KLD = -0.5 * tf.reduce_sum(1 + tf.log(tf.pow(sigma0,2)) - tf.pow(mu0,2) - tf.pow(sigma0,2))
                for i in range(sigma1._shape[1]):
                    self.KLD += -0.5 * tf.reduce_sum(1 + tf.log(tf.pow(sigma1[:,i],2)) - tf.pow(mu1[:,i], 2) - tf.pow(sigma1[:,i], 2))

            # for idx, dist_info in enumerate(self.model.reg_latent_dist.split_dist_info(fake_reg_z_dist_info)):
            #     if "stddev" in dist_info:
            #         self.log_vars.append(("max_std_%d" % idx, tf.reduce_max(dist_info["stddev"])))
            #         self.log_vars.append(("min_std_%d" % idx, tf.reduce_min(dist_info["stddev"])))
            #
            # self.log_vars.append(("MI", mi_est))
            # self.log_vars.append(("CrossEnt", cross_ent))

            all_vars = tf.trainable_variables()
            d_vars = [var for var in all_vars if var.name.startswith('d_')]
            c_vars = [var for var in all_vars if var.name.startswith('c_')]
            v_vars = [var for var in all_vars if var.name.startswith('v_')]
            g_vars = [var for var in all_vars if var.name.startswith('g_')]

            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))

            classifer_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
            self.classifer_trainer = pt.apply_optimizer(classifer_optimizer, losses=[classifier_loss], var_list=d_vars+c_vars)

            vae_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
            self.vae_trainer = pt.apply_optimizer(vae_optimizer, losses=[self.R1+self.KLD], var_list=g_vars+v_vars+d_vars)
            self.log_vars.append(("KLD", tf.reduce_mean(self.KLD)))
            self.log_vars.append(("vae", tf.reduce_mean(self.R1+self.KLD)))

            # discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
            discriminator_optimizer = tf.train.GradientDescentOptimizer(self.discriminator_learning_rate)
            self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer, losses=[discriminator_loss],var_list=d_vars)

            # generator_optimizer = tf.train.GradientDescentOptimizer(self.generator_learning_rate)
            generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
            self.generator_trainer = pt.apply_optimizer(generator_optimizer, losses=[generator_loss], var_list=g_vars)

            # R_optimizer = tf.train.GradientDescentOptimizer(self.discriminator_learning_rate)
            R_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate/50., beta1=0.5)
            # self.R_trainer = pt.apply_optimizer(R_optimizer, losses=[self.R1*10.+self.R0])
            self.R1_trainer = pt.apply_optimizer(R_optimizer, losses=[self.R1])

            for k, v in self.log_vars:
                tf.summary.scalar(k, v)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                # self.visualize_all_factors()
                print('testing visual')

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

            if self.pretrain_classifier:
                for epoch in range(400):
                    for i in range(20):
                        x, y = self.dataset.supervised_train.next_batch(self.batch_size)
                        feed_dict = {self.input_tensor: x, self.input_label: y}
                        if epoch<100: #%5 == 0:
                            sess.run(self.classifer_trainer, feed_dict)
                        else:
                            sess.run(self.vae_trainer, feed_dict)
                        summary_str = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary_str, counter)
                        counter += 1
                fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, 'Classifier'))
                print("Classifier saved in file: %s" % fn)

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                all_log_vals = []
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    x, y = self.dataset.supervised_train.next_batch(self.batch_size)
                    if self.has_classifier:
                        feed_dict = {self.input_tensor: x, self.input_label: y}
                    else:
                        feed_dict = {self.input_tensor: x}
                    log_vals = sess.run([self.discriminator_trainer] + log_vars, feed_dict)[1:]
                    sess.run(self.generator_trainer, feed_dict)
                    sess.run(self.R_trainer, feed_dict)
                    all_log_vals.append(log_vals)
                    counter += 1

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                x, y = self.dataset.supervised_train.next_batch(self.batch_size)
                if self.has_classifier:
                    feed_dict = {self.input_tensor: x, self.input_label: y}
                else:
                    feed_dict = {self.input_tensor: x}
                summary_str = sess.run(summary_op, feed_dict)
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_dict = dict(zip(log_keys, avg_log_vals))

                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")


    def visualize_all_factors(self):
        if self.model.network_type=='rec_crs' or self.model.network_type=='rec_crs2':
            with tf.Session():
                z_var = []
                z_var_reduce = []
                if CLASS==3:
                    for cat in [ [0, 1, 1,], [1, 0, 0,], [0, 1, 0], [0, 0, 1, ], [1, 1, 0,]]: # only the first categorical val is useful
                        for nregidx in [0, 1]:
                            for contidx in range(0, 10, 1):
                                cont = contidx / 10.
                                z_var_reduce = z_var_reduce + [[nregidx] + cat + [cont]]
                    _, x_dist_info = self.model.generate(z_var_reduce)
                    img_var = x_dist_info["p"]
                    img_var = self.dataset.inverse_transform(img_var)
                    rows = 10
                    img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
                    img_var = img_var[:rows * rows, :, :, :]
                    imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
                else:
                    for cat in [[1, 0], [0, 1], [1, 1,],[0,0]]:  # only the first categorical val is useful
                        for nregidx in [0, 1/3., 1]:
                            for contidx in range(0, 10, 1):
                                cont = contidx / 10.
                                z_var_reduce = z_var_reduce + [[nregidx] + cat + [cont]]
                    _, x_dist_info = self.model.generate(z_var_reduce)
                    img_var = x_dist_info["p"]
                    img_var = self.dataset.inverse_transform(img_var)
                    rows = 10
                    img_var = tf.reshape(img_var, [120] + list(self.dataset.image_shape))
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
                tf.summary.image("image__rectcrs",imgs)  # Hope: this should be changed into 3D
                self.imgs = imgs
        else:
            with tf.Session():
                fixed_noncat = np.concatenate([
                    np.tile(
                        self.model.nonreg_latent_dist.sample_prior(10).eval(),
                        [10, 1]
                    ),
                    self.model.nonreg_latent_dist.sample_prior(self.batch_size - 100).eval(),
                ], axis=0)
                fixed_cat = np.concatenate([
                    np.tile(
                        self.model.reg_latent_dist.sample_prior(10).eval(),
                        [10, 1]
                    ),
                    self.model.reg_latent_dist.sample_prior(self.batch_size - 100).eval(),
                ], axis=0)

            offset = 0
            for dist_idx, dist in enumerate(self.model.reg_latent_dist.dists):
                if isinstance(dist, Gaussian):
                    if self.model.network_type == "ModelNet": #don't need this data for modelnet
                        continue
                    assert dist.dim == 1, "Only dim=1 is currently supported"
                    c_vals = []
                    for idx in xrange(10):
                        c_vals.extend([-1.0 + idx * 2.0 / 9] * 10)
                    c_vals.extend([0.] * (self.batch_size - 100))
                    vary_cat = np.asarray(c_vals, dtype=np.float32).reshape((-1, 1))
                    cur_cat = np.copy(fixed_cat)
                    cur_cat[:, offset:offset+1] = vary_cat
                    offset += 1
                elif isinstance(dist, Categorical):
                    lookup = np.eye(dist.dim, dtype=np.float32)
                    cat_ids = []
                    for idx in xrange(10):
                        cat_ids.extend([idx] * 10)
                    cat_ids.extend([0] * (self.batch_size - 100))
                    cur_cat = np.copy(fixed_cat)
                    cur_cat[:, offset:offset+dist.dim] = lookup[cat_ids]
                    offset += dist.dim
                elif isinstance(dist, Bernoulli):
                    if self.model.network_type == "ModelNet":
                        continue
                    assert dist.dim == 1, "Only dim=1 is currently supported"
                    lookup = np.eye(dist.dim, dtype=np.float32)
                    cat_ids = []
                    for idx in xrange(10):
                        cat_ids.extend([int(idx / 5)] * 10)
                    cat_ids.extend([0] * (self.batch_size - 100))
                    cur_cat = np.copy(fixed_cat)
                    cur_cat[:, offset:offset+dist.dim] = np.expand_dims(np.array(cat_ids), axis=-1)
                    # import ipdb; ipdb.set_trace()
                    offset += dist.dim
                else:
                    raise NotImplementedError
                z_var = tf.constant(np.concatenate([fixed_noncat, cur_cat], axis=1))

                _, x_dist_info = self.model.generate(z_var)

                # just take the mean image
                if isinstance(self.model.output_dist, Bernoulli):
                    img_var = x_dist_info["p"]
                elif isinstance(self.model.output_dist, Gaussian):
                    img_var = x_dist_info["mean"]
                else:
                    raise NotImplementedError
                img_var = self.dataset.inverse_transform(img_var)
                rows = 10
                img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
                img_var = img_var[:rows * rows, :, :, :]
                imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))

                if self.model.network_type == "mnist":
                    stacked_img = []
                    for row in xrange(rows):
                        row_img = []
                        for col in xrange(rows):
                            row_img.append(imgs[row, col, :, :, :])
                        stacked_img.append(tf.concat(1, row_img))
                    imgs = tf.concat(0, stacked_img)
                    imgs = tf.expand_dims(imgs, 0)
                    tf.summary.image("image_%d_%s" % (dist_idx, dist.__class__.__name__), imgs) # Hope: this should be changed into 3D
                    self.imgs = imgs
                elif self.model.network_type == "ModelNet":
                    self.imgs = imgs
                #     imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
                #     if isinstance(dist, Categorical):
                #         saver = tf.train.Saver({"gen_imgs": imgs})


    def generating(self):
        self.init_opt()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # load model
            model_name = '/home/hope-yao/Documents/InfoGAN/ckt/rec_crs2/rec_crs2_2017_02_05_21_01_56/Classifier.ckpt.meta'
            saver = tf.train.Saver()
            new_saver = tf.train.import_meta_graph(model_name)
            saver.restore(sess, '/home/hope-yao/Documents/InfoGAN/ckt/rec_crs2/rec_crs2_2017_02_05_21_01_56/Classifier.ckpt')

            a = np.asarray([[x / 10., y / 10.] for x in range(10) for y in range(10)])
            b = np.tile([1, 0], (100, 1))
            c = np.concatenate((a[:, 0].reshape(100, 1), b, a[:, 1].reshape(100, 1), np.zeros((100,10))), axis=1)

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
            model_name = '/home/hope-yao/Documents/InfoGAN/ckt/rec_crs2/rec_crs2_2017_02_05_00_10_50/Classifier.ckpt.meta'
            saver = tf.train.Saver()
            new_saver = tf.train.import_meta_graph(model_name)
            saver.restore(sess, '/home/hope-yao/Documents/InfoGAN/ckt/rec_crs2/rec_crs2_2017_02_05_00_10_50/Classifier.ckpt')

            input_tensor, input_label = self.dataset.supervised_train.next_batch(self.batch_size)

            real_d, _, real_reg_z_dist_info, real_reg_dist_flat = self.model.discriminate(input_tensor)

            s1 = real_reg_z_dist_info['id_1_prob']
            s2 = real_reg_z_dist_info['id_2_prob']

            mean = real_reg_z_dist_info['id_0_mean']
            stddev = real_reg_z_dist_info['id_0_stddev']
            epsilon = tf.random_normal(tf.shape(mean))
            s0 = mean + epsilon * stddev

            mean = real_reg_z_dist_info['id_3_mean']
            stddev = real_reg_z_dist_info['id_3_stddev']
            epsilon = tf.random_normal(tf.shape(mean))
            s4 = mean + epsilon * stddev

            z_var_reduce = tf.concat(1, [tf.reshape(s0[:, 0], (self.batch_size, 1)),
                                         tf.reshape(s1[:, 0], (self.batch_size, 1)),
                                         tf.reshape(s2[:, 0], (self.batch_size, 1)),
                                         tf.reshape(s4[:, 0], (self.batch_size, 1)), ])
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
            print('%0.16f' % real_d.eval())
            print('%0.16f' % real_reg_z_dist_info['id_0_prob'].eval()[0][0])
            print('%0.16f' % real_reg_z_dist_info['id_1_prob'].eval()[0][0])
            if CLASS==3:
                print('%0.16f' % real_reg_z_dist_info['id_2_prob'].eval()[0][0])
                print('%0.16f' % real_reg_z_dist_info['id_3_mean'].eval())
            else:
                print('%0.16f' % real_reg_z_dist_info['id_2_mean'].eval())
            print('done')

    def classify(self):
        self.init_opt()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # load model
            model_name = '/home/hope-yao/Documents/InfoGAN/ckt/rec_crs/rec_crs_2017_01_30_12_49_23/rec_crs_2017_01_30_12_49_23_9000.ckpt.meta'
            saver = tf.train.Saver()
            new_saver = tf.train.import_meta_graph(model_name)
            saver.restore(sess, '/home/hope-yao/Documents/InfoGAN/ckt/rec_crs/rec_crs_2017_01_30_12_49_23/rec_crs_2017_01_30_12_49_23_9000.ckpt')

            img_in, input_label = self.dataset.supervised_train.next_batch(self.batch_size)

            real_d, _, real_reg_z_dist_info, _ = self.model.discriminate(img_in)
            tt = self.model.disc_reg_dist_info(real_reg_z_dist_info)
            tt0 = tt['id_0_prob']
            tt1 = tt['id_1_prob']
            if CLASS==3:
                tt2 = tt['id_2_prob']
                prediction = tf.concat(1, [tf.reshape(tt0[:, 0], (self.batch_size, 1)), tf.reshape(tt1[:, 0], (self.batch_size, 1)),
                                           tf.reshape(tt2[:, 0], (self.batch_size, 1))])
            else:
                prediction = tf.concat(1, [tf.reshape(tt0[:, 0], (self.batch_size, 1)), tf.reshape(tt1[:, 0], (self.batch_size, 1))])

            print(prediction.eval())
            print(input_label - prediction.eval())
            classifier_loss = -tf.reduce_sum(tf.log(
                tf.reduce_sum(tf.multiply(prediction, input_label) + tf.multiply((1 - prediction), (1 - input_label)),
                              axis=1)))
            print(classifier_loss.eval())


            z_var = np.concatenate([
                np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]*10).reshape(self.batch_size, 1),
                tt0[:, 0].eval().reshape(self.batch_size, 1),
                tt1[:, 0].eval().reshape(self.batch_size, 1),
                tt2[:, 0].eval().reshape(self.batch_size, 1),
                real_reg_z_dist_info['id_3_prob'].eval().reshape(self.batch_size, 1)
            ], axis = 1)
            print(input_label)
            print(z_var)
            fake_x, _ = self.model.generate(z_var.tolist())
            img = np.asarray(fake_x.eval()).reshape(self.batch_size, 28, 28)
            import matplotlib.pyplot as plt
            plt.imshow(img[0])
            plt.show()
            plt.hold(True)
