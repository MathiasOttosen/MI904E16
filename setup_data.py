import csv
from skimage import io
from skimage.color import rgb2gray
import os


# This method returns the filenames in the ImageCollection
def get_filenames(images, test_images=None):
    names = []
    for s in images.files:
        # Since the filenames in the ImageCollection is relative paths
        # we split the string and return the last part of the string
        # i.e. the actual filename
        names.append(str.split(s, '/')[-1])
    if test_images is not None:
        test_names = get_filenames(test_images)
        return(names, test_names)
    return(names)


def get_target_values(labels, filenames, test_filenames=None):
    targets = []
    for s in filenames:
        targets.append(labels[s])
    if test_filenames is not None:
        test_targets = get_target_values(labels, test_filenames)
        return(targets, test_targets)
    return(targets)


# A method that crops an image at the center by crop_width
def center_crop(image, crop_width):
    cropped_image = image[image.shape[0]/2 - crop_width/2:
                          image.shape[0]/2 + crop_width/2,
                          image.shape[1]/2 - crop_width/2:
                          image.shape[1]/2 + crop_width/2]
    return(cropped_image)


# A load function to be used with ImageCollection
# It uses the standard imread but grayscales them first
def imread_gray(f, img_num):
    return rgb2gray(io.imread(f))


def imread_crop(f, img_num):
    return(center_crop(io.imread(f), 224))


# A method for loading a collection of images from the disk
# The func parameter is the load function that is used in the ImageCollection
def load_images(location, func=io.imread):
    def load_train_images(location, func):
        return(io.ImageCollection(os.path.join(location + 'train_samples/*.jpg'),
                                  load_func=func))

    def load_test_images(location, func):
        return(io.ImageCollection(os.path.join(location + 'test_samples/*.jpg'),
                                  load_func=func))
    return(load_train_images(location, func),
           load_test_images(location, func))


# A method for loading the labels into a dictionary
def load_labels(labels='metadata/train_info.csv'):
    with open(labels) as csvfile:
        # We use the csv.DictReader object to format the
        # csvfile into a dictionary
        reader = csv.DictReader(csvfile)
        reader_dict = {}
        # Here we iterate over the DictReader and put the
        # contents into a Python dict object
        for index, row in enumerate(reader):
            reader_dict[row['filename']] = row['style']
        return(reader_dict)


def setup_data(location):
    images, test_images = load_images(location)
    names, test_names = get_filenames(images, test_images=test_images)
    labels = load_labels()
    targets, test_targets = get_target_values(labels,
                                              names, test_filenames=test_names)
    return(images, test_images, targets, test_targets)
