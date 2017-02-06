from infogan.misc.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
import prettytensor as pt
import tensorflow as tf
import infogan.misc.custom_ops
from infogan.misc.custom_ops import leaky_rectify


class RegularizedGAN(object):
    def __init__(self, output_dist, latent_spec, batch_size, image_shape, network_type):
        """
        :type output_dist: Distribution
        :type latent_spec: list[(Distribution, bool)]
        :type batch_size: int
        :type network_type: string
        """
        self.output_dist = output_dist
        self.latent_spec = latent_spec
        self.latent_dist = Product([x for x, _ in latent_spec])
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        self.network_type = network_type
        self.image_shape = image_shape
        assert all(isinstance(x, (Gaussian, Categorical, Bernoulli)) for x in self.reg_latent_dist.dists)

        self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        self.reg_disc_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

        image_size = image_shape[0]
        if network_type == "ModelNet":
            with tf.variable_scope("d_net"):
                shared_template = \
                    (pt.template("input").
                     reshape([-1] + list(image_shape)).
                     # custom_conv2d(64, k_h=4, k_w=4).
                     custom_conv3d(64, k_h=4, k_w=4, k_d=4).
                     apply(leaky_rectify).
                     # custom_conv2d(128, k_h=4, k_w=4).
                     custom_conv3d(128, k_h=4, k_w=4, k_d=4).
                     conv_batch_norm_3d().
                     apply(leaky_rectify).
                     custom_fully_connected_3d(1024).
                     fc_batch_norm_3d().
                     apply(leaky_rectify))
                self.discriminator_template = shared_template.custom_fully_connected_3d(1)
                self.encoder_template = \
                    (shared_template.
                     custom_fully_connected_3d(128).
                     fc_batch_norm_3d().
                     apply(leaky_rectify).
                     custom_fully_connected_3d(self.reg_latent_dist.dist_flat_dim))

            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected_3d(1024).
                     fc_batch_norm_3d().
                     apply(tf.nn.relu).
                     custom_fully_connected_3d(image_size / 4 * image_size / 4  * image_size / 4 * 128).
                     fc_batch_norm_3d().
                     apply(tf.nn.relu).
                     # reshape([-1, image_size / 4, image_size / 4, 128]).
                     # custom_deconv2d([0, image_size / 2, image_size / 2, 64], k_h=4, k_w=4).
                     reshape([-1, image_size / 4, image_size / 4, image_size / 4, 128]).
                     custom_deconv3d([0, image_size / 2, image_size / 2, image_size / 2, 64], k_h=4, k_w=4, k_d=4).
                     conv_batch_norm_3d().
                     apply(tf.nn.relu).
                     # custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     custom_deconv3d([0] + list(image_shape), k_h=4, k_w=4, k_d=4).
                     flatten())
        elif network_type == "mnist" or network_type == "rec_crs" or network_type == "rec_crs2":
            with tf.variable_scope("d_net"):
                shared_template = \
                    (pt.template("input").
                     reshape([-1] + list(image_shape)).
                     custom_conv2d(64, k_h=4, k_w=4).
                     apply(leaky_rectify).
                     custom_conv2d(128, k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(leaky_rectify).
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(leaky_rectify))
                self.discriminator_template = shared_template.custom_fully_connected(1)
                self.encoder_template = \
                    (shared_template.
                     custom_fully_connected(128).
                     fc_batch_norm().
                     apply(leaky_rectify).
                     custom_fully_connected(self.latent_dist.dist_flat_dim))  # modified by Hope, also output reconstruction of noise variables
                self.encoder_template_all = \
                    (shared_template.
                     custom_fully_connected(128).
                     fc_batch_norm().
                     apply(leaky_rectify))

            with tf.variable_scope("c_net"):
                self.classifier = \
                    (self.encoder_template_all.
                     custom_fully_connected(2*2))

            with tf.variable_scope("v_net"):
                self.variations = \
                    (self.encoder_template_all.
                     custom_fully_connected(2*2))

            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(image_size / 4 * image_size / 4 * 128).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size / 4, image_size / 4, 128]).
                     custom_deconv2d([0, image_size / 2, image_size / 2, 64], k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     flatten())
        # elif network_type == "rec_crs":
        #     with tf.variable_scope("d_net"):
        #         HID_DIM1 = 3
        #         inputs = pt.template("input")
        #         hidden1 = inputs.reshape([-1] + list(image_shape)).conv2d(4,2).reshape([-1] + [28 * 28 * 2])
        #         hidden21 = hidden1[:, 0:28 * 28]
        #         hidden22 = hidden1[:, 28 * 28:28 * 28 * 2]
        #         hidden31 = hidden21.fully_connected(image_size * image_size + 100)
        #         hidden32 = hidden22.fully_connected(image_size * image_size + 100)
        #         hidden40 = hidden31[:, 0:100] + hidden32[:, 0:100]
        #         hidden41 = hidden31[:, 100:image_size * image_size]
        #         hidden42 = hidden32[:, 100:image_size * image_size]
        #         hidden50 = hidden40.fully_connected(1)
        #         hidden51 = hidden41.fully_connected(1)
        #         hidden52 = hidden42.fully_connected(1)
        #
        #         shared_template = hidden50.join([hidden51, hidden52])
        #
        #         self.discriminator_template = shared_template.custom_fully_connected(1)
        #         self.encoder_template = \
        #             (shared_template.
        #              custom_fully_connected(128).
        #              fc_batch_norm().
        #              apply(leaky_rectify).
        #              custom_fully_connected(self.reg_latent_dist.dist_flat_dim))
        #
        #     with tf.variable_scope("g_net"):
        #         HID_DIM1 = 2
        #         all_inputs = pt.template("input")
        #         pretty_input_cat = all_inputs[:, 1:1+HID_DIM1]  # pt.template("input_cat")
        #         pretty_input_cont = all_inputs[:, 1+HID_DIM1:]  # pt.template("input_cont")
        #
        #         seq_cnt = pretty_input_cont.fully_connected(100)
        #
        #         seq_cat = []
        #         rotated_output = []
        #         output = []
        #         for i in range(HID_DIM1):
        #             tmp = pretty_input_cat[:, i].reshape([self.batch_size, 1])
        #             seq_cat = seq_cat + [tmp.fully_connected(image_size * image_size)]
        #             #         seq_cat[i].reshape(([-1, image_size, image_size, 1]))
        #
        #             rotated_output = rotated_output + [seq_cnt.join([seq_cat[i]])]
        #             print(i)
        #             output = output + [rotated_output[i].fully_connected(image_size * image_size)]
        #
        #         self.generator_template = output[0]
        #         for i in range(1, HID_DIM1):
        #             self.generator_template += output[i]
        else:
            raise NotImplementedError

    def discriminate(self, x_var):
        d_out = self.discriminator_template.construct(input=x_var)
        d = tf.nn.sigmoid(d_out[:, 0])
        c = self.classifier.construct(input=x_var)
        z = self.variations.construct(input=x_var)
        reg_dist_flat = tf.concat(1, [z[:,0:2],c,z[:,2:4]])
        # reg_dist_flat = self.encoder_template.construct(input=x_var)
        reg_dist_info = self.latent_dist.activate_dist(reg_dist_flat)
        return d, self.latent_dist.sample(reg_dist_info), reg_dist_info, reg_dist_flat

    def generate(self, z_var):
        x_dist_flat = self.generator_template.construct(input=z_var)
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)
        return self.output_dist.sample(x_dist_info), x_dist_info

    def disc_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(z_i)
        return self.reg_disc_latent_dist.join_vars(ret)

    def cont_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, Gaussian):
                ret.append(z_i)
        return self.reg_cont_latent_dist.join_vars(ret)

    def disc_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(dist_info_i)
        return self.reg_disc_latent_dist.join_dist_infos(ret)

    def cont_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, Gaussian):
                ret.append(dist_info_i)
        return self.reg_cont_latent_dist.join_dist_infos(ret)

    def reg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if reg_i:
                ret.append(z_i)
        return self.reg_latent_dist.join_vars(ret)

    def nonreg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if not reg_i:
                ret.append(z_i)
        return self.nonreg_latent_dist.join_vars(ret)

    def reg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if reg_i:
                ret.append(dist_info_i)
        return self.reg_latent_dist.join_dist_infos(ret)

    def nonreg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if not reg_i:
                ret.append(dist_info_i)
        return self.nonreg_latent_dist.join_dist_infos(ret)

    def combine_reg_nonreg_z(self, reg_z_var, nonreg_z_var):
        reg_z_vars = self.reg_latent_dist.split_var(reg_z_var)
        reg_idx = 0
        nonreg_z_vars = self.nonreg_latent_dist.split_var(nonreg_z_var)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_z_vars[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_z_vars[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_vars(ret)

    def combine_reg_nonreg_dist_info(self, reg_dist_info, nonreg_dist_info):
        reg_dist_infos = self.reg_latent_dist.split_dist_info(reg_dist_info)
        reg_idx = 0
        nonreg_dist_infos = self.nonreg_latent_dist.split_dist_info(nonreg_dist_info)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_dist_infos[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_dist_infos[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_dist_infos(ret)