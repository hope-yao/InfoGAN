import numpy as np
from tensorflow.examples.tutorials import mnist
import os
import numpy as np


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
        categorical_labels = np.zeros((labels.shape[0], 10))
        for idx, label in enumerate(labels):
            cat_label = np.zeros(10)
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


class MnistDataset(object):
    def __init__(self,switch_categorical_label):
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
            sup_images.extend(self.train.images[ids])
            sup_labels.extend(self.train.labels[ids])
        np.random.set_state(rnd_state)
        self.supervised_train = supervised_Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
            switch_categorical_label=switch_categorical_label
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class ModelNet10(object):
    def __init__(self,switch_categorical_label):
        # data_directory = "ModelNet10"
        from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
        from tensorflow.python.framework import dtypes
        dtype = dtypes.float32
        reshape = False

        from fuel.datasets.hdf5 import H5PYDataset
        train_set = H5PYDataset('ModelNet10/ModelNet10.hdf5', which_sets=('train',))
        handle = train_set.open()
        train_data = train_set.get_data(handle, slice(0, train_set.num_examples))
        train_images = train_data[0]
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[2] * train_images.shape[3] * train_images.shape[4])
        train_labels = train_data[1]

        validation_size = int(0.1*train_set.num_examples)
        validation_images = train_images[:validation_size]
        validation_labels = train_labels[:validation_size]
        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]

        test_set = H5PYDataset('ModelNet10/ModelNet10.hdf5', which_sets=('test',))
        handle = test_set.open()
        test_data = test_set.get_data(handle, slice(0, test_set.num_examples))
        test_images = test_data[0]
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[2] * test_images.shape[3] * test_images.shape[4])
        test_labels = test_data[1]

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
        self.image_dim = 32 * 32 * 32
        self.image_shape = (32, 32, 32, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class rec_crs(object):
    def __init__(self,switch_categorical_label):

        data = np.load('rec_crs.npy')
        data0 = data.item()['rectcrs_img']
        data1 = data.item()['rect_img']
        data2 = data.item()['cross_img']
        data3 = data.item()['sph_img']
        self.label = label = [[[1, 1, 0]], [[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]]
        # self.label = label = [[[1,0, 1,0, 0,1]], [[1,0, 0,1, 0,1]], [[0,1, 1,0, 0,1]], [[0,1, 0,1, 1,0]]]
        label0 = np.asarray(label[0]*data0.shape[0])
        label1 = np.asarray(label[1]*data1.shape[0])
        label2 = np.asarray(label[2]*data2.shape[0])
        label3 = np.asarray(label[3]*data3.shape[0])
        # label0 = np.asarray([[1,0,0]]*data0.shape[0])
        # label1 = np.asarray([[0,1,0]]*data1.shape[0])
        # label2 = np.asarray([[0,0,1]]*data2.shape[0])

        images = np.concatenate([data0, data1, data2, data3])
        labels = np.concatenate([label0, label1, label2, label3])
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

