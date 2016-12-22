import numpy as np
from math import sqrt
from skimage import feature
from skimage.color import rgb2gray, rgb2hsv, gray2rgb
from skimage.transform import probabilistic_hough_line as phl
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def run_experiments(images, test_images, targets, test_targets):
    with open('results', 'a') as r, open('debug', 'a') as d:
        clf = MultinomialNB()
        try:
            npzfile = np.load('features.npz')
            line_features = nan_to_num(fix_input_from_file(npzfile['line_features'], targets))
            color_features = nan_to_num(fix_input_from_file(npzfile['color_features'], targets))
            test_line_features = nan_to_num(fix_input_from_file(npzfile['test_line_features'], test_targets))
            test_color_features = nan_to_num(fix_input_from_file(npzfile['test_color_features'], test_targets))
            error_pics = npzfile['error_pics']
            test_error_pics = npzfile['test_error_pics']
        except IOError:
            line_features, color_features, error_pics = extract_features(images, d)
            test_line_features, test_color_features, test_error_pics = extract_features(test_images, d)
            np.savez('features', line_features=line_features, color_features=color_features,
                     test_line_features=test_line_features, test_color_features=test_color_features,
                     error_pics=error_pics, test_error_pics=test_error_pics)
        for l in error_pics:
            p = l[1]
            p.reverse()
            for pics in p:
                del targets[l[0]][pics]
        for l in test_error_pics:
            p = l[1]
            p.reverse()
            for pics in p:
                del test_targets[l[0]][pics]
        predictions = []
        for i in range(len(images)):
            _ = clf.fit(line_features[i], targets[i])
            predictions = clf.predict(test_line_features[i])
            r.write('Line experiment ' + str(i+1) +
                    '\n Precision score: ' + str(precision_score(test_targets[i], predictions, average='weighted')) +
                    '\n Recall Score: ' + str(recall_score(test_targets[i], predictions, average='weighted')) +
                    '\n F1 score: ' + str(f1_score(test_targets[i], predictions, average='weighted')) + '\n')
            r.flush()
            _ = clf.fit(color_features[i], targets[i])
            predictions = clf.predict(test_color_features[i])
            r.write('Color experiment ' + str(i+1) +
                    '\n Precision score: ' + str(precision_score(test_targets[i], predictions, average='weighted')) +
                    '\n Recall Score: ' + str(recall_score(test_targets[i], predictions, average='weighted')) +
                    '\n F1 score: ' + str(f1_score(test_targets[i], predictions, average='weighted')) + '\n')
            _ = clf.fit(np.concatenate((line_features[i], color_features[i]), axis=1), targets[i])
            r.flush()
            predictions = clf.predict(np.concatenate((test_line_features[i], test_color_features[i]), axis=1))
            r.write('Combined experiment ' + str(i+1) +
                    '\n Precision score: ' + str(precision_score(test_targets[i], predictions, average='weighted')) +
                    '\n Recall Score: ' + str(recall_score(test_targets[i], predictions, average='weighted')) +
                    '\n F1 score: ' + str(f1_score(test_targets[i], predictions, average='weighted')) + '\n')
            r.flush()
        line_features = concatenate_arrays(line_features)
        color_features = concatenate_arrays(color_features)
        test_line_features = concatenate_arrays(test_line_features)
        test_color_features = concatenate_arrays(test_color_features)
        targets = concatenate_arrays(targets)
        test_targets = concatenate_arrays(test_targets)
        _ = clf.fit(line_features, targets)
        predictions = clf.predict(test_line_features)
        r.write('Full line experiment' +
                '\n Precision score: ' + str(precision_score(test_targets, predictions, average='weighted')) +
                '\n Recall Score: ' + str(recall_score(test_targets, predictions, average='weighted')) +
                '\n F1 score: ' + str(f1_score(test_targets, predictions, average='weighted')) + '\n')
        r.flush()
        _ = clf.fit(color_features, targets)
        predictions = clf.predict(test_color_features)
        r.write('Full color experiment' +
                '\n Precision score: ' + str(precision_score(test_targets, predictions, average='weighted')) +
                '\n Recall Score: ' + str(recall_score(test_targets, predictions, average='weighted')) +
                '\n F1 score: ' + str(f1_score(test_targets, predictions, average='weighted')) + '\n')
        r.flush()
        _ = clf.fit(np.concatenate((line_features, color_features), axis=1), targets)
        predictions = clf.predict(np.concatenate((test_line_features, test_color_features), axis=1))
        r.write('Full combined experiment' +
                '\n Precision score: ' + str(precision_score(test_targets, predictions, average='weighted')) +
                '\n Recall Score: ' + str(recall_score(test_targets, predictions, average='weighted')) +
                '\n F1 score: ' + str(f1_score(test_targets, predictions, average='weighted')) + '\n')
        r.flush()
        r.write('Total amount of paintings:\n Training: ' + str(len(targets)) +
                '\n Test: ' + str(len(test_targets)) + '\n')
        r.flush()


def make_confusion_matrices(line_features, color_features, test_line_features,
                            test_color_features, targets, test_targets,
                            label_order, step=5):
    def build_conf_mat(features, test_features, targets, test_targets, c,
                       order):
        clf = MultinomialNB()
        _ = clf.fit(features, targets)
        pred = clf.predict(test_features)
        c.write(str(confusion_matrix(test_targets, pred, labels=order)) + '\t')
    with open('conf_mat', 'a') as c:
        for i in range(len(line_features)):
            build_conf_mat(np.nan_to_num(line_features[i]), np.nan_to_num(test_line_features[i]),
                           targets[i], test_targets[i], c, label_order[i*step:step*(i+1)])
            c.write('\n\n')
        for i in range(len(color_features)):
            build_conf_mat(np.nan_to_num(color_features[i]), np.nan_to_num(test_color_features[i]),
                           targets[i], test_targets[i], c, label_order[i*step:step*(i+1)])
            c.write('\n\n')
        for i in range(len(line_features)):
            build_conf_mat(np.nan_to_num(np.concatenate((line_features[i], color_features[i]), axis=1)),
                           np.nan_to_num(np.concatenate((test_line_features[i], test_color_features[i]), axis=1)),
                           targets[i], test_targets[i], c, label_order[i*step:step*(i+1)])
            c.write('\n\n')
        line_features = concatenate_arrays(line_features)
        color_features = concatenate_arrays(color_features)
        test_line_features = concatenate_arrays(test_line_features)
        test_color_features = concatenate_arrays(test_color_features)
        targets = concatenate_arrays(targets)
        test_targets = concatenate_arrays(test_targets)
        build_conf_mat(np.nan_to_num(line_features), np.nan_to_num(test_line_features),
                       targets, test_targets, c, label_order)
        c.write('\n\n')
        build_conf_mat(np.nan_to_num(color_features), np.nan_to_num(test_color_features),
                       targets, test_targets, c, label_order)
        c.write('\n\n')
        build_conf_mat(np.nan_to_num(np.concatenate((line_features, color_features), axis=1)),
                       np.nan_to_num(np.concatenate((test_line_features, test_color_features), axis=1)),
                       targets, test_targets, c, label_order)
        c.write('\n\n')


def fix_input_from_file(i, actual):
    start = len(actual[0])
    for l in range(1, len(i)):
        i[l] = i[l][start:]
        start += len(actual[l])
    return(i)


def concatenate_arrays(arr):
    return(np.concatenate((arr), axis=0))


def extract_features(images, d):
    line_features = []
    color_features = []
    colors, lines = [], []
    error_pics = []
    indices = []
    for group, i in enumerate(images):
        lines, color = [], []
        for index, image in enumerate(i):
            if len(image.shape) < 2:
                indices.append(index)
                d.write('(' + str(group) + ', ' + str(index) + ' bad image)\n')
                continue
            l = get_line_features(image)
            lines.append(l)
            c = get_color_features(image)
            colors.append(c)
            d.write('(' + str(group) + ', ' + str(index) + ', ' + str(l) + ', ' + str(c) + ')\n')
        line_features.append(np.vstack(lines))
        color_features.append(np.vstack(colors))
        if len(indices) != 0:
            error_pics.append((group, indices))
    line_features = np.nan_to_num(line_features)
    color_features = np.nan_to_num(color_features)
    return(line_features, color_features, error_pics)


def get_line_features(image):
    return(construct_line_features(image))


def nan_to_num(features):
    for i in range(len(features)):
        features[i] = np.nan_to_num(features[i])
    return(features)


def construct_line_features(image):
    lines = extract_straight_lines(image)
    avg_num_lines = float(len(lines))/(image.shape[0] * image.shape[1])
    line_lengths = compute_line_lengths(lines)
    avg_line_length = np.mean(line_lengths)
    max_line = np.max(line_lengths) if len(line_lengths) != 0 else 0
    length_dist = max_line - avg_line_length
    line_features = [avg_line_length, avg_num_lines, length_dist]
    return(line_features)


def extract_straight_lines(image):
    s_lines = []
    rgb_image = rgb2gray(image) if len(image.shape) == 3 else image
    canny = feature.canny(rgb_image, sigma=4)
    s_lines.extend(phl(canny, line_length=20, line_gap=5))
    return(s_lines)


def line_length(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    l = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return(l)


def compute_line_lengths(lines):
    lengths = [line_length(l[0], l[1]) for l in lines]
    return(lengths)


def make_hue_histogram(hsv_img, c=1):
    hue = (hsv_img[:, :, 0] * 360).round()
    bins = 6 * c
    hist, edges = np.histogram(hue, bins=bins, range=(0.0, 360.0), density=True)
    hist = hist*np.diff(edges)
    return(hist, edges)


def get_top_N_values_in_hist(hist, n):
    tmp_hist = hist
    top = []
    index = []
    i = 0
    while i < n:
        t = tmp_hist.max()
        idx = np.where(hist == t)[0]
        index.extend(idx)
        tmp_hist = np.delete(tmp_hist, idx)
        i += 1
    if len(index) > n:
        index = index[:n]
    top.extend(hist[index])
    return(np.sort(top), np.sort(index))


def get_color_features(image):
    return(construct_color_features(image))


def construct_color_features(image):
    features = []
    if len(image.shape) == 3:
        if (image.shape)[2] == 3:
            hsv_img = rgb2hsv(image)
        else:
            gray_img = rgb2gray(image)
            rgb_img = gray2rgb(gray_img)
            hsv_img = rgb2hsv(rgb_img)
    else:
        rgb_img = gray2rgb(image)
        hsv_img = rgb2hsv(rgb_img)
    hist, edges = make_hue_histogram(hsv_img, c=4)
    top = get_top_N_values_in_hist(hist, 5)[0]
    sat = np.mean(hsv_img[:, :, 1])
    val = np.mean(hsv_img[:, :, 2])
    f = [sat, val]
    f.extend(top)
    f.append(top[-1] - np.mean(top))
    features.extend(f)
    return(features)
