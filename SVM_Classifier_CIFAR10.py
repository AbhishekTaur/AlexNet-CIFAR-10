import numpy as np
import os
import sys
import tarfile
import shutil
from six.moves import urllib
from sklearn.svm import SVC
from tqdm import tqdm
import time


DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

data_dir = '/tmp/cifar10_data'

train_dir = '/tmp/cifar10_train'


def download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_files(filenames):
    batch_images = []
    labels = []
    images = []
    for file in filenames:
        data = unpickle(file)
        labels = labels + data[b'labels']
        batch_images.append(data[b'data'])

    for batch in batch_images:
        for d in batch:
            images.append(d)

    labels = np.asarray(labels)
    labels = labels.astype(np.uint8)

    return images, labels


def get_data(data_dir):
    filenames = [os.path.join(data_dir, 'data_batch_%d' % i)
                 for i in range(1, 6)]
    for f in filenames:
        if not os.path.exists(data_dir):
            raise ValueError('Failed to find file: ' + f)

    return read_files(filenames=filenames)


def train(data_dir, classifier):
    if not os.path.exists(data_dir):
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(data_dir, 'cifar-10-batches-py')

    X, y = get_data(data_dir=data_dir)

    tqdm(classifier.fit(X, y))
    print('train accuracy: ', classifier.score(X, y))


def test(data_dir, classifier):
    if not os.path.exists(data_dir):
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    X, y = read_files([os.path.join(data_dir, 'test_batch')])
    print('test accuracy: ', classifier.score(X, y))


def main():
    start_time = time.time()
    download_and_extract()
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)
    svm_clf = SVC(kernel="linear", C=float("inf"))
    train(data_dir=data_dir, classifier=svm_clf)
    test(data_dir=data_dir, classifier=svm_clf)
    print(time.time() - start_time)


if __name__ == '__main__':
    main()
