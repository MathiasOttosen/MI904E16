import csv
from skimage import io
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


def setup_data(location, labels=None):
    images, test_images = load_images(location)
    names, test_names = get_filenames(images, test_images=test_images)
    labels = load_labels() if labels is None else labels
    targets, test_targets = get_target_values(labels,
                                              names, test_filenames=test_names)
    return(images, test_images, targets, test_targets)


def split_classes(location, labels, classes, step=5, factor=0.25):
    images, test_images = load_images(location)

    def make_image_list(classes, labels, images, step, factor):
        d = dict()
        for c in classes:
            d[c] = []
        for s in images.files:
            d[labels[str.split(s, '/')[-1]]].append(s)
        i_list = []
        c = 0
        while c < len(classes):
            names = []
            for n in classes[c: c + step]:
                style = d[n]
                names.extend(style[:int(factor*len(style))])
            ic = io.ImageCollection(names, conserve_memory=True)
            i_list.append(ic)
            c += step
        return(i_list)

    files = make_image_list(classes, labels, images, step, factor)
    test_files = make_image_list(classes, labels, test_images, step, factor)
    names, targets, test_targets = [], [], []
    for l in files:
        names = get_filenames(l)
        targets.append(get_target_values(labels, names))
    for l in test_files:
        names = get_filenames(l)
        test_targets.append(get_target_values(labels, names))
    return(files, test_files, targets, test_targets)
