import cv2
import tensorflow as tf
import csv
import numpy as np

DATASET_PATH = '/media/andrea/New Volume/train/train'
TRAIN_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/100_train_subset.csv'
TEST_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/100_valid_subset.csv'
VALID_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/100_test_subset.csv'

BATCH_SIZE = 30
NUM_OF_CLASSES = 5

def decode_csv(csv_path):
    f = open(csv_path, 'rt')

    img_paths = []
    img_labels = []
    # read csv file
    try:
        reader = csv.reader(f)
        next(reader)
        for row in reader:

            img_paths.append(row[0])
            img_labels.append(row[1])

    finally:
        f.close()

    return img_paths, np.array(img_labels, dtype=np.int8)

# code from Mnist example https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
class DataSet(object):

    def __init__(self,
                 images_paths,
                 labels,
                 one_hot=False,
                 #dtype=tf.dtypes.float32,
                 reshape=True,
                 preprocess=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """

        self._num_examples = len(images_paths)
        self._images_paths = np.array(images_paths)
        self._labels = np.array(labels)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._one_hot = one_hot
        self._reshape = reshape
        self._preprocess = preprocess

        if self._one_hot:
            self._labels = self.prepare_labels(self._labels)
            print(self._labels.get_shape())
            #print(np.shape(self._labels))

        # # Convert shape from [num examples, rows, columns, depth]
        # # to [num examples, rows*columns] (assuming depth == 1)
        # if reshape:
        #   assert images.shape[3] == 1
        #   images = images.reshape(images.shape[0],
        #                           images.shape[1] * images.shape[2])
        # if dtype == dtypes.float32:
        #   # Convert from [0, 255] -> [0.0, 1.0].
        #   images = images.astype(numpy.float32)
        #   images = numpy.multiply(images, 1.0 / 255.0)


    @property
    def images_paths(self):
        return self._images_paths

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def one_hot(self):
        return self._one_hot

    @property
    def reshape(self):
        return self._reshape

    @property
    def preprocess(self):
        return self._preprocess

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
            self._images_paths = self._images_paths[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        images = self.prepare_images(self._images_paths[start:end])
        return images, self._labels[start:end]

    def prepare_labels(self, labels):
        one_hot_labels = []
        for l in labels:
            one_hot_label = np.zeros(NUM_OF_CLASSES, dtype=np.int8)
            one_hot_label[l] = 1
            one_hot_labels.append(one_hot_label)
        return tf.pack(one_hot_labels)
        #return np.array(one_hot_labels, dtype=np.int8)

    def center_crop(self, image, desired_height=256, desired_width=256):

        ratio = desired_width / float(desired_height)

        image_shape = tf.shape(image)

        # if we have grayscale images
        image = tf.reshape(image, [image_shape[0], image_shape[1], 1])

        height = tf.cast(image_shape[0], tf.float32)
        width = tf.cast(image_shape[1], tf.float32)

        condition = tf.greater(tf.mul(height, ratio), width)

        target_width = tf.cast(tf.select(condition, width, tf.mul(height, ratio)), tf.int32)
        target_height = tf.cast(tf.select(condition, tf.div(height, ratio), height), tf.int32)

        # could also do resizing first

        image = tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)

        # we want input for nn to be small numbers
        image = tf.image.resize_images(image, (desired_height, desired_width)) / 255.0

        return image

    def prepare_images(self, img_paths):
        images = []
        for p in img_paths:
            image = cv2.imread(p,0)
            if self._preprocess:
                image = self.center_crop(image=image)

            images.append(image)

        images = tf.pack(images)
        print(images.get_shape())
        if self._reshape:
            images = tf.reshape(images, [BATCH_SIZE, 256*256])
            print(images.get_shape())
        return images

img_paths, img_labels = decode_csv(TRAIN_PATH)
train_data = DataSet(img_paths, img_labels, one_hot=True, reshape=True)


# part of code from https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 256*256])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_OF_CLASSES])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# because input in conv2d must have shape [batch, width, height, channels]
# btw this doesn't make sense if the reshape is False, then he reshapes it again, silly
x_image = tf.reshape(x, [-1,256,256,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([122 * 122 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 122*122*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, NUM_OF_CLASSES])
b_fc2 = bias_variable([NUM_OF_CLASSES])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = train_data.next_batch(BATCH_SIZE)
  imgs, lbls = sess.run([batch[0], batch[1]])
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:imgs, y_: lbls, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: imgs, y_: lbls, keep_prob: 0.5})

