import csv
import shutil
from skimage import io
from skimage.color import rgb2gray


# This method returns the filenames in the ImageCollection
def get_filenames(images):
    names = []
    for s in images.files:
        # Since the filenames in the ImageCollection is relative paths
        # we split the string and return the last part of the string
        # i.e. the actual filename
        names.append(str.split(s, '/')[-1])
    return(names)


# A load function to be used with ImageCollection
# It uses the standard imread but grayscales them first
def imread_gray(f, img_num):
    return rgb2gray(io.imread(f))


# A method for loading a collection of images from the disk
# The func parameter is the load function that is used in the ImageCollection
def load_images(location='train/', func=io.imread):
    # We use the io.ImageCollection object with conserve_memory=True
    return(io.ImageCollection(location + '*.jpg',
                              conserve_memory=True, load_func=func))


# A method for copying specific files from
# one directory into another for training and test
# With the limit parameter we can limit the amount of images that are copied
def copy_samples(filenames, limit=0, src='train/',
                 train_dst='train_samples/', test_dst='test_samples/'):
    # First we make another limiter for the test data,
    # which is 10 percent of the training data
    if limit is not 0:
        test_limit = int(round(limit * 0.1))
    else:
        test_limit = int(round(len(filenames) * 0.1))

    # First we copy the test images and then the training images
    for s in filenames[:test_limit-1]:
        shutil.copy2(src + s, test_dst)
    for s in filenames[test_limit:limit-1]:
        shutil.copy2(src + s, train_dst)


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


# A method for getting all the filenames for a particular style
def find_filenames_by_style(style, labels=None):
    if labels is None:
        labels = load_labels()

    fnames = []
    # We iterate over all the key-value pairs in label
    # and find the filenames that has the style
    for t in labels.items():
        if t[1] == style:
            fnames.append(t[0])
            continue
        else:
            continue

    return(fnames)


# A class for handling the features extracted from the images
# Because the images would be too much to have in memory
# this container makes use of the ImageCollection object and
# applies the extraction method givin by the extract paramater
class FeatureCollection:
    # The class is instantiated with an ImageCollection and an extraction method
    def __init__(self, coll, extract):
        self.image_collection = coll
        self.extraction_method = extract

    def __len__(self):
        return(len(self.image_collection))

    # Here we apply the extraction method on the image returned from the
    # ImageCollection
    def __getitem__(self, key):
        return(self.extraction_method(self.image_collection[key]))
