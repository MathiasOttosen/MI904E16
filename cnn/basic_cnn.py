import cv2
import tensorflow as tf
import csv
import numpy as np
import tensorflow.contrib.slim as slim
import sklearn.metrics

DATASET_PATH = '/media/andrea/New Volume/train/train'
TRAIN_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/final_train.csv'
TEST_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/final_test.csv'

BATCH_SIZE = 40
NUM_OF_CLASSES = 25
CROP = 256

LOAD_VARS = False
TRAIN = True
TEST = True

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

    def resize(self, image, desired_height=256, desired_width=256):

        image = cv2.resize(image, (desired_height, desired_width))
        image = np.reshape(image, [np.shape(image)[0] * np.shape(image)[1]])
        image = np.multiply(image, 1.0 / 255.0)

        return image

    def prepare_images(self, img_paths, batch_size):
        images = []
        for p in img_paths:
            image = cv2.imread(p,0)
            if self._preprocess:
                image = self.resize(image=image, desired_width=CROP, desired_height=CROP)
            images.append(image)
        return np.array(images)

img_paths, img_labels = decode_csv(TRAIN_PATH)
train_data = DataSet(img_paths, img_labels, one_hot=True, reshape=True)

# part of code from https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, CROP*CROP])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_OF_CLASSES])

# because input in conv2d must have shape [batch, width, height, channels]
# btw this doesn't make sense if the reshape is False, then he reshapes it again, silly
x_image = tf.reshape(x, [-1,CROP,CROP,1])

FILTER = 10
STRIDE = 5
NUM_OF_MAX_POOL = 2
CHANNELS = 1
#OUTPUT_SIZE = CROP/2**NUM_OF_MAX_POOL
OUTPUT_SIZE = 3

def network(net):
    with tf.variable_scope('Network'), slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
        net = slim.repeat(net, 2, slim.conv2d, 8, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 16, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, scope='fc6')
        net = slim.fully_connected(net, NUM_OF_CLASSES, activation_fn=None, scope='fc8')

        return net

def network1(net):
    with tf.variable_scope('Network'), slim.arg_scope([slim.conv2d],activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                      weights_regularizer=slim.l2_regularizer(0.0005)):
        dims = 8  # bilo 48
        net = slim.conv2d(net, dims / 2, 7, 2)
        net = slim.max_pool2d(net,[2,2])
        net = slim.conv2d(net, dims, 5)
        net = slim.max_pool2d(net, [2,2])

        net = slim.conv2d(net, dims, 3)
        net = slim.conv2d(net, dims, 3)
        net = slim.conv2d(net, dims, 3)

        net = slim.flatten(net)

        net = slim.fully_connected(net, 25)
        net = slim.fully_connected(net, NUM_OF_CLASSES, activation_fn=None)

        return net

y_conv = network1(x_image)

saver = tf.train.Saver()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = 0
processed_examples = 0

LOAD_MODEL_PATH = ""

if LOAD_VARS:
    saver.restore(sess, LOAD_MODEL_PATH)
else:
    sess.run(tf.initialize_all_variables())

SAVE_MODEL_PATH = ""
if TRAIN:
    accs = []
    losses = []
    #14570 iters = 10 epohs
    for i in range(14570):
      imgs, lbls = train_data.next_batch(BATCH_SIZE)
      processed_examples += BATCH_SIZE
      if processed_examples >= train_data.num_examples:
          processed_examples -= train_data.num_examples
          epochs += 1
          print("EPOCH" + str(epochs) + " REACHED")
          save_path = saver.save(sess, SAVE_MODEL_PATH)
          print("Model saved in file: %s" % save_path)
      if i%1000 == 0:
          save_path = saver.save(sess, SAVE_MODEL_PATH)
          print("Model saved in file: %s" % save_path)
      if True:
          train_accuracy, loss = sess.run([accuracy, cross_entropy], feed_dict={x:imgs, y_: lbls})
          accs.append(train_accuracy)
          losses.append(loss)
          print("step %d, training accuracy %g"%(i, train_accuracy))
          print("step %d, loss %g"%(i, loss))
      train_step.run(feed_dict={x: imgs, y_: lbls})

    with open('accuracies2.txt', 'a') as f:
        for a in accs:
            f.write("%s\n" % a)

    with open('losses2.txt', 'a') as f:
        for l in losses:
            f.write("%s\n" % l)


TEST_DIVISION = 463
if TEST:
    img_paths_test, img_labels_test = decode_csv(TEST_PATH)
    test_data = DataSet(img_paths_test, img_labels_test, one_hot=True, reshape=True)

    TEST_BATCH_SIZE = test_data.num_examples / TEST_DIVISION

    y_p = tf.argmax(y_conv, 1)
    y_true = img_labels_test
    test_accs = []
    y_predicted = []
    for i in range(TEST_DIVISION):
        test_batch = test_data.next_batch(TEST_BATCH_SIZE)
        test_acc, y_pred, y_bla = sess.run([accuracy, y_p, tf.nn.softmax(y_conv)],feed_dict={x: test_batch[0], y_: test_batch[1]})
        y_predicted.extend(y_pred)
        print("%i Test accuracy %g"% (i,test_acc))

        test_accs.append(test_acc)

    precision = sklearn.metrics.precision_score(y_true, y_predicted, average='weighted')
    recall = sklearn.metrics.recall_score(y_true, y_predicted, average='weighted')
    f1_score = sklearn.metrics.f1_score(y_true, y_predicted, average='weighted')
    print("ACCURACY " + str(sum(test_accs)/TEST_DIVISION))
    print("PRECISION " + str(precision))
    print("RECALL " + str(recall))
    print("F1 " + str(f1_score))
    print sklearn.metrics.confusion_matrix(y_true, y_predicted)
