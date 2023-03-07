import glob
import tensorflow as tf
import numpy as np

class DataLoader():
    def __init__(self, data_path, split):
        assert split in ['train', 'test', 'val'], \
            'ERROR: mode should be either test, train or val.'

        self.filenames = glob.glob(f'{data_path}/{split}/*.tfrec')
        self.split = split
        if self.split in ['train', 'val']:
            self.label_name = 'class'
            self.label_dtype = tf.int64
        else:
            self.label_name = 'id'
            self.label_dtype = tf.string
        self.data = self.read_tfrec()
        self.len = self.get_dataset_len()

    def __len__(self):
        return self.len

    def get_dataset_len(self):
        for i, r in self.data.enumerate():
            pass
        return(i.numpy()+1)

    def read_tfrec(self):
        tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            self.label_name: tf.io.FixedLenFeature([], self.label_dtype),
        }

        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False

        dataset = tf.data.TFRecordDataset(self.filenames).with_options(ignore_order)
        parse_img_fn = lambda x: tf.io.parse_single_example(x, tfrec_format)
        dataset = dataset.map(parse_img_fn)

        return(dataset)

    def batch(self, batch_size):
        batches = self.data.shuffle( buffer_size=self.len ).batch(batch_size)
        