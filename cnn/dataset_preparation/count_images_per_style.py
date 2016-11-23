# makes new csv file with all dataset images - 1st column filepath, 2nd column style name
import csv
import operator


CSV_DATASET_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/dataset.csv'
COUNT_CSV_FILE_PATH = '/home/andrea/Documents/project/MI904E16/cnn/dataset_files/style_count.csv'

# CSV_DATASET_PATH = '/home/andrea/Documents/project/dataset.csv'
# COUNT_CSV_FILE_PATH = '/home/andrea/Documents/project/style_count.csv'

f = open(CSV_DATASET_PATH, 'rt')

# dict of styles and number of images
style_dict = {}

try:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        style = row[1]
        if style not in style_dict:
            style_dict[style] = 1
        else:
            style_dict[style] += 1
finally:
    f.close()

# sort dictionary in descending order
sorted_dict = sorted(style_dict.items(), key=operator.itemgetter(1), reverse=True)

# write the dict values in new csv file with two columns: style name, number of images
# descending order
f = open(COUNT_CSV_FILE_PATH, 'wt')
try:
    writer = csv.writer(f)
    writer.writerow(('style', 'count'))
    sum = 0
    for i in sorted_dict:
        sum += i[1]
        writer.writerow((i[0], i[1]))

finally:
    f.close()

#print(sum)