import numpy as np
from tensorflow.examples.tutorials import mnist
import os
import numpy as np


class Dataset(object):
    def __init__(self, images, labels=None):
        self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = images.shape[0]
        # shuffle on first run
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            return self._images[start:end], self._labels[start:end]


class MnistDataset(object):
    def __init__(self):
        data_directory = "MNIST"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class supervised_Dataset(object):
    def __init__(self, images, labels=None, switch_categorical_label=False ):
        self._images = images.reshape(images.shape[0], -1)
        self._switch_categorical_label = switch_categorical_label
        if switch_categorical_label:
            self._labels = self.convert_label(labels) # Modified by Hope
        else:
            self._labels = labels
        self._epochs_completed = -1
        self._num_examples = images.shape[0]
        # shuffle on first run
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def convert_label(self, labels):
        categorical_labels = np.zeros((labels.shape[0], labels.max()))
        for idx, label in enumerate(labels):
            cat_label = np.zeros(labels.max())
            cat_label[label] = 1
            categorical_labels[idx] = cat_label
        return categorical_labels

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            if self._switch_categorical_label:
                return self._images[start:end], self._labels[start:end,:]
            else:
                return self._images[start:end], self._labels[start:end]


class rec_crs(object):
    def __init__(self,switch_categorical_label):

        data = np.load('rec_crs.npy')
        data1 = data.item()['rect_img']
        data2 = data.item()['cross_img']
        label1 = np.zeros(data1.shape[0])
        label2 = np.ones(data2.shape[0])
        # label1 = data.item()['rect_label']
        # label2 = data.item()['cross_label']
        # label1 = np.append(label1, np.zeros((label1.shape[0], 8)), 1)
        # label2 = np.append(label2, np.zeros((label2.shape[0], 8)), 1)

        images = np.concatenate([data1, data2])
        labels = np.concatenate([label1, label2])
        labels = np.uint8(labels)
        images = images.reshape(images.shape[0],28,28,1)

        from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
        from tensorflow.python.framework import dtypes
        dtype = dtypes.float32
        reshape = True


        from random import sample
        ttnum = len(images)
        trainval_num = int(0.9*ttnum)
        trainval_num_idx = sample(range(ttnum), trainval_num )

        test_images = np.delete(images, trainval_num_idx, 0)
        test_labels = np.delete(labels, trainval_num_idx, 0)
        trainval_images = images[trainval_num_idx]
        trainval_labels = labels[trainval_num_idx]

        train_num_idx = sample(range(trainval_num), int(0.8*trainval_num))
        validation_images = np.delete(trainval_images, train_num_idx, 0)
        validation_labels = np.delete(trainval_labels, train_num_idx, 0)
        train_images = trainval_images[train_num_idx]
        train_labels = trainval_labels[train_num_idx]

        self.train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
        self.validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)
        self.test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids])
            sup_labels.extend(self.train.labels[ids])
        np.random.set_state(rnd_state)
        self.supervised_train = supervised_Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
            switch_categorical_label = switch_categorical_label
        )
        self.image_dim = 28* 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

