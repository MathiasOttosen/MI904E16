import csv
import os
import shutil
from skimage import io


# A method that loads the file names of all images in the
# location and sorts them alphabetically
def load_filenames(location='train/'):
    # First we list all files in the directory
    lst = os.listdir(location)
    # This first sorts by lowest string value
    # then by length of the string
    lst.sort(key=str.lower)
    lst.sort(key=str.__len__)
    return(lst)


# A method for loading images from the disk
def load_images(location='train/'):
    # We use the io.ImageCollection object with conserve_memory=True
    return(io.ImageCollection(location + '*.jpg',
                              conserve_memory=True))


# A method for copying specific files from one directory into another
# With the limit parameter we can limit the amount of images that are copied
def copy_samples(filenames, limit=0, src='train/', dst='samples/'):
    for s in filenames[:limit-1]:
        shutil.copy2(src + s, dst)


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
