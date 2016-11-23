# makes new csv file with all dataset images - 1st column filepath, 2nd column style name
import csv
import os.path
import glob

# CSV_FILE_PATH = '/home/andrea/Documents/project/train_info.csv'
# NEW_CSV_FILE_PATH = '/home/andrea/Documents/project/dataset.csv'

CSV_FILE_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/train_info.csv'
NEW_CSV_FILE_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/dataset.csv'

TRAIN_PATH = '/media/andrea/New Volume/train/train'

# we don't want to include images with errors
errorish_imgs = [os.path.basename(x) for x in glob.glob("/home/andrea/Documents/project/incorrect images/error/still errorish/*.jpg")]

f = open(CSV_FILE_PATH, 'rt')

#list of images' filepaths and corresponding styles
files_styles = []

#read csv file, make new list with values: [filepath, style index]
try:
    reader = csv.reader(f)
    next(reader)
    for row in reader:

        if row[0] in errorish_imgs:
            print(row[0] + " doesnt exist!")
            continue

        file_path = TRAIN_PATH + '/' + row[0]
        style = row[3]
        files_styles.append([file_path, style])

finally:
    f.close()

#write the list values in new csv file with two columns: filepath, style name
f = open(NEW_CSV_FILE_PATH, 'wt')
try:
    writer = csv.writer(f)
    writer.writerow(('filepath', 'style'))
    for i in files_styles:
        writer.writerow((i[0], i[1]))

finally:
    f.close()