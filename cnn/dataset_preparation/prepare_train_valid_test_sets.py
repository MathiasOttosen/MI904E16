# makes csv files with images for train, validation and test
from __future__ import print_function

import csv
import hashlib


SUBSET_CSV_FILE_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/100_subset.csv'
CSV_TRAIN_FILE_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/100_train_subset.csv'
CSV_VALID_FILE_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/100_valid_subset.csv'
CSV_TEST_FILE_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/100_test_subset.csv'



USE_FAST_HASH = True  # hash only the path, not the file content, faster

train = []
validation = []
test = []

def hash_me(img_path):

    if USE_FAST_HASH:
        img_bytes = img_path
    else:
        with open(img_path) as f:
            img_bytes = f.read()
        #more precise but slower variant is:
        # img_bytes = cv2.imread(img_path).tostring()


    precision = 4

    hex_hash = hashlib.sha1(img_bytes).hexdigest()[:precision]

    percentage = int(hex_hash, base=16) / float(16 ** precision)
    return percentage

with open(SUBSET_CSV_FILE_PATH, 'rt') as f:
    reader = csv.reader(f)
    next(reader)

    for i,row in enumerate(reader):
        if (i+1)%100 == 0:
            print('Processed %d files'%(i + 1))

        file_path = row[0]
        style = row[1]

        hash_percentage = hash_me(file_path)

        if hash_percentage < 0.7:
            train.append((file_path, style))
        elif hash_percentage < 0.85:
            validation.append((file_path, style))
        else:
            test.append((file_path, style))

for example_list, path in zip([train,validation,test],[CSV_TRAIN_FILE_PATH,CSV_VALID_FILE_PATH,CSV_TEST_FILE_PATH]):
    with open(path,'w') as f:
        writer = csv.writer(f)
        writer.writerow(('filepath', 'style index'))
        for example in example_list:
            writer.writerow(example)

        print('Written %d files to: %s'%(len(example_list),path))