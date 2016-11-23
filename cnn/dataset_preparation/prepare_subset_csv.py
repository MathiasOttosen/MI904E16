# makes a new csv file with filepaths and corresponding style indices and style names for chosen classes
from __future__ import print_function

import csv
import os.path as osp

CSV_FILE_PATH = '/home/andrea/PycharmProjects/automated-art-expert/dataset_files/dataset.csv'
SUBSET_CSV_FILE_PATH = '/home/andrea/PycharmProjects/automated-art-expert/dataset_files/dataset_subset.csv'

# CSV_FILE_PATH = '/home/andrea/Documents/project/dataset.csv'
# SUBSET_CSV_FILE_PATH = '/home/andrea/Documents/project/dataset_subset.csv'

# impressionism, cubism, pop art, realism, northern renaissance
chosen_classes = ["Impressionism", "Cubism", "Realism", "Pop Art", "Northern Renaissance"]
class_idx_dict = {}

for i in range(len(chosen_classes)):
    class_idx_dict[chosen_classes[i]] = i

# list of images' filepaths, corresponding style names
files_styles = []

# read csv file, make new list with values: [filepath, style name] ONLY for chosen classes
with open(CSV_FILE_PATH, 'rt') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        file_path = row[0]
        style = row[1]

        if style in chosen_classes:
            if not osp.exists(file_path):
                print('[W] File ignored, not found on FS: %s'%file_path)
            else:
                files_styles.append([file_path, style])

# write the list values in new csv file with three columns: filepath, style index, style name
with open(SUBSET_CSV_FILE_PATH, 'wt') as f:
    writer = csv.writer(f)
    writer.writerow(('filepath', 'style index', 'style name'))
    for i in files_styles:
        writer.writerow((i[0], class_idx_dict[i[1]], i[1]))
