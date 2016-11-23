# normalize in a way that there is an equal amount of imgs for every class
from __future__ import print_function

import csv
import random
from collections import defaultdict


# CSV_TRAIN_FILE_PATH = '/home/andrea/Documents/project/train_subset.csv'
# CSV_TRAIN_NORMALIZED_FILE_PATH = '/home/andrea/Documents/project/train_subset_normalized.csv'

CSV_TRAIN_FILE_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/train_subset.csv'
CSV_TRAIN_NORMALIZED_FILE_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/train_subset_normalized.csv'

# read csv file, make new list with values: [filepath, style name] ONLY for chosen classes
with open(CSV_TRAIN_FILE_PATH, 'rt') as f:
    reader = csv.reader(f)
    next(reader)

    dataset = defaultdict(list)
    [dataset[img_cls].append(img_path) for img_path, img_cls in reader]

lengths ={key:len(value) for key,value in dataset.iteritems()}
max_length = max(lengths.values())
min_length = min(lengths.values())


dataset = {key:(value*int(max_length/float(len(value))+1))[:max_length] for key,value in dataset.iteritems()}

print(map(len,dataset.values()))

example_list = [(image_path,example_cls) for example_cls, image_paths in dataset.iteritems() for image_path in image_paths]


random.seed(69)
random.shuffle(example_list)

with open(CSV_TRAIN_NORMALIZED_FILE_PATH, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(('filepath', 'style index'))
    for example in example_list:
        writer.writerow(example)

    print('Written %d files to: %s' % (len(example_list), CSV_TRAIN_NORMALIZED_FILE_PATH))