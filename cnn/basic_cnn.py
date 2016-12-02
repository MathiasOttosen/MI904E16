import cv2
import tensorflow as tf
import csv
import numpy as np

DATASET_PATH = '/media/andrea/New Volume/train/train'
TRAIN_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/600_train_subset.csv'
VALID_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/600_valid_subset.csv'
TEST_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/600_test_subset.csv'

BATCH_SIZE = 50
NUM_OF_CLASSES = 5
CROP = 28

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
            #print(self._labels.get_shape())
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
            perm = np.arange(self._num_examples, dtype=np.int32)
            np.random.shuffle(perm)
            self._images_paths = self._images_paths[perm]
            #print(self._labels.get_shape())
            #print(type(perm))
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        images = self.prepare_images(self._images_paths[start:end], batch_size)
        labels = self._labels[start:end]
        return images, labels

    def prepare_labels(self, labels):
        one_hot_labels = []
        for l in labels:
            one_hot_label = np.zeros(NUM_OF_CLASSES, dtype=np.int32)
            one_hot_label[l] = 1
            one_hot_labels.append(one_hot_label)
        #return tf.pack(one_hot_labels)
        return np.array(one_hot_labels, dtype=np.int32)

    def center_crop(self, image, desired_height=256, desired_width=256):

        # could also do resizing first
        image = cv2.resize(image, (desired_height, desired_width))
        image = np.reshape(image, [np.shape(image)[0] * np.shape(image)[1]])
        image = np.multiply(image, 1.0 / 255.0)

        # we want input for nn to be small numbers
        #image = tf.image.resize_images(image, (desired_height, desired_width)) / 255.0

        return image

    def prepare_images(self, img_paths, batch_size):
        images = []
        for p in img_paths:
            image = cv2.imread(p,0)
            if self._preprocess:
                image = self.center_crop(image=image, desired_width=CROP, desired_height=CROP)
            images.append(image)
        return np.array(images)

img_paths, img_labels = decode_csv(TRAIN_PATH)
train_data = DataSet(img_paths, img_labels, one_hot=True, reshape=True)


# part of code from https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, CROP*CROP])
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


def convolutional_layer(input, stride, channels, output_depth):
    W_conv = weight_variable([stride, stride, channels, output_depth])
    b_conv = bias_variable([output_depth])

    h_conv = tf.nn.relu(conv2d(input, W_conv) + b_conv)

    return h_conv

def max_pool_layer(input):
    return max_pool_2x2(input)

# because input in conv2d must have shape [batch, width, height, channels]
# btw this doesn't make sense if the reshape is False, then he reshapes it again, silly
x_image = tf.reshape(x, [-1,CROP,CROP,1])

conv1 = convolutional_layer(x_image, 5, 1, 32)
maxpool1 = max_pool_layer(conv1)

conv2 = convolutional_layer(maxpool1, 5, 32, 64)
maxpool2 = max_pool_layer(conv2)

# conv3 = convolutional_layer(maxpool2, 5, 16, 32)
# maxpool3 = max_pool_layer(conv3)
#
# conv4 = convolutional_layer(maxpool3, 5, 32, 64)
# maxpool4 = max_pool_layer(conv4)
#
# conv5 = convolutional_layer(maxpool4, 5, 64, 128)
# maxpool5 = max_pool_layer(conv5)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(maxpool2, [-1, 7*7*64])
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
for i in range(1000):
  imgs, lbls = train_data.next_batch(BATCH_SIZE)
  #if i%100 == 0:
  if True:
    train_accuracy = accuracy.eval(feed_dict={
        x:imgs, y_: lbls, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: imgs, y_: lbls, keep_prob: 0.5})


img_paths_test, img_labels_test = decode_csv(TEST_PATH)
test_data = DataSet(img_paths_test, img_labels_test, one_hot=True, reshape=True)
test_batch = test_data.next_batch(test_data.num_examples)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

