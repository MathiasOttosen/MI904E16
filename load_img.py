import csv
import os
from skimage import io

training_location = '/home/replacedleaf60/SW9/Training data/train/'
label = '/home/replacedleaf60/SW9/Training data/metadata/train_info.csv'


# A method that loads the file names of all images in the
# training_location and sorts them alphabetically
def load_filenames():
    # First we list all files in the directory
    lst = os.listdir(training_location)
    # This first sorts by lowest string value
    # then by length of the string
    lst.sort(key=(str.lower | str.__len__))
    return(lst)


# A method for loading images from the disk
def load_images():
    # We use the io.ImageCollection object with conserve_memory=True
    return(io.ImageCollection(training_location + '*.jpg',
                              conserve_memory=True))


# A method for loading the labels into a dictionary
def load_labels():
    with open(label) as csvfile:
        # We use the csv.DictReader object to format the
        # csvfile into a dictionary
        reader = csv.DictReader(csvfile)
        reader_dict = {}
        # Here we iterate over the DictReader and put the
        # contents into a Python dict object
        for index, row in enumerate(reader):
            reader_dict[row['filename']] = row['style']
        return(reader_dict)
