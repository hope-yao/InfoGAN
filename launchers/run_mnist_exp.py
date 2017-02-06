from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import os
from infogan.misc.datasets import MnistDataset, ModelNet10, rec_crs, rec_crs2
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime

if __name__ == "__main__":

    network_type = "rec_crs2"
    switch_categorical_label = True # Modified by Hope, for supervised learning

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/" + network_type
    root_checkpoint_dir = "ckt/" + network_type
    batch_size = 100
    updates_per_epoch = 100
    max_epoch = 400

    exp_name =  network_type + "_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    if network_type == "mnist":
        dataset = MnistDataset(switch_categorical_label)
        latent_spec = [
            (Uniform(1), False),
            (Categorical(10), True),
            (Uniform(1, fix_std=True), True),
            (Uniform(1, fix_std=True), True),
        ]
    elif network_type == "ModelNet":
        dataset = ModelNet10(switch_categorical_label)
        latent_spec = [
            (Uniform(1), False),
            (Categorical(10), True),
            (Uniform(1, fix_std=True), True),
            # (Uniform(1, fix_std=True), True),
        ]
    elif network_type == "rec_crs":
        dataset = rec_crs(False)
        latent_spec = [
            (Uniform(1), False),
            (Categorical(2), True),
            (Categorical(2), True),
            (Categorical(2), True),
            (Uniform(1, fix_std=True), True),
        ]
    elif network_type == "rec_crs2":
        dataset = rec_crs2(False)
        latent_spec = [
            (Uniform(1), False),
            (Categorical(2), True),
            (Categorical(2), True),
            (Uniform(1, fix_std=True), True),
            # (Uniform(10), False),
        ]
    else:
        raise NotImplementedError

    model = RegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type=network_type,
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        # generator_learning_rate=1e-4,
        # discriminator_learning_rate=2e-5,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
        has_classifier = True,
        pretrain_classifier = True,
    )

    algo.train()
    # algo.regen()
    # algo.classify()


