def unpack_dataset(data_path):
    import glob
    import numpy as np
    from PIL import Image
    import tensorflow as tf
    import os
    import pandas as pd

    if not os.path.exists('data'):
        os.mkdir('data')

    for split in ['train', 'val', 'test']:
        filenames = glob.glob(f'{data_path}/{split}/*.tfrec')
        if split in ['train', 'val']:
            tfrec_format = {
                "image": tf.io.FixedLenFeature([], tf.string),
                'class': tf.io.FixedLenFeature([], tf.int64),
                'id': tf.io.FixedLenFeature([], tf.string),
           }
        else:
            tfrec_format = {
                "image": tf.io.FixedLenFeature([], tf.string),
                'id': tf.io.FixedLenFeature([], tf.string),
           }

        dataset = tf.data.TFRecordDataset(filenames)
        parse_img_fn = lambda x: tf.io.parse_single_example(x, tfrec_format)
        dataset = dataset.map(parse_img_fn)

        idx = []
        classes = []
        filenames_jpg = []

        for sample in dataset.enumerate():
            id_val = sample[1]['id'].numpy().decode('utf-8')
            if split in ['train', 'val']:
                class_val = sample[1]['class'].numpy()
            else:
                class_val = None
            img = tf.image.decode_jpeg(
                sample[1]['image'], channels=3).numpy()
            img = Image.fromarray(img)
            filename = f'{id_val}.jpg'

            idx.append(id_val)
            classes.append(class_val)
            filenames_jpg.append(filename)

            img.save(f'data/{filename}')

        df = pd.DataFrame({'id': idx, 'filename':filenames_jpg, 'class': classes})
        df.to_csv(f'data/{split}.csv')


def get_data(url, path, filename):
    from urllib.request import urlretrieve
    import os

    # filename = f"{filename}.{url.split('.')[-1]}"
    
    if filename not in os.listdir(path):
        print('Downloading the dataset ...')
        urlretrieve(url, f'{path}/{filename}')
        print('Dataset downloaded.')
    else:
        print('Dataset is already present.')