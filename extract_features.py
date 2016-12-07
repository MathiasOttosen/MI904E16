import numpy as np
from math import sqrt, pow
from skimage import feature
from skimage.color import rgb2gray, rgb2hsv, gray2rgb
from skimage.transform import probabilistic_hough_line as phl


src = 'experiments/tweak/experiment_1/'


def get_line_features(images, test_images=None):
    features = []
    n = 0
    for i in images:
        features.append(construct_line_features(i))
        n += 1
    features = np.vstack(features)
    if test_images is not None:
        test_features = get_line_features(test_images)
        return(features, test_features)
    else:
        return(features)


def construct_line_features(image):
    lines = extract_straight_lines(image)
    mean_straight_length = compute_avg_straght_length(lines)
    line_features = [mean_straight_length]
    return(line_features)


def segment_image(image, h_segs, v_segs):
    height = image.shape[0] / h_segs
    width = image.shape[1] / v_segs
    segments = []
    for i in range(1, h_segs+1):
        for n in range(1, v_segs+1):
            seg = image[height * (i-1):height * i, width * (n-1):width * n]
            segments.append(seg)
    return(segments)


def line_length(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    l = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    return(l)


def compute_avg_straght_length(lines):
    lengths = [line_length(l[0], l[1]) for l in lines]
    return(np.mean(lengths))


def extract_straight_lines(image):
    s_lines = []
    canny = feature.canny(rgb2gray(image))
    s_lines.extend(phl(canny))
    return(s_lines)


def make_hue_histogram(hsv_img, c=1):
    hue = (hsv_img[:, :, 0] * 360).round()
    bins = 5 * c
    hist, edges = np.histogram(hue, bins=bins, range=(0.0, 300.0), density=True)
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
        if t == 0 and n != 1:
            index.append(idx[0])
            i += 1
            continue
        elif t == 1 and n != 1:
            index.extend(idx)
            i += 1
            while i < n:
                idx += 1
                if idx >= len(hist):
                    idx = 0
                index.extend(idx)
                i += 1
            break
        if len(idx) != 1:
            i += len(idx)
        index.extend(idx)
        tmp_hist = np.delete(tmp_hist, idx)
        i += 1
    top.extend(hist[index])
    return(np.sort(top), np.sort(index))


def construct_color_features(images, test_images=None):
    features = []
    for i in images:
        if len(i.shape) == 3:
            hsv_img = rgb2hsv(i)
        else:
            rgb_img = gray2rgb(i)
            hsv_img = rgb2hsv(rgb_img)
        hist, edges = make_hue_histogram(hsv_img, c=2)
        top = get_top_N_values_in_hist(hist, 3)[0]
        sat = np.mean(hsv_img[:, :, 1])
        val = np.mean(hsv_img[:, :, 2])
        f = [sat, val]
        f.extend(top)
        # f = extract_color_from_segmented_image(hsv_img)
        features.append(f)
    features = np.round(np.vstack(features), decimals=3)
    if test_images is not None:
        test_features = construct_color_features(test_images)
        return(features, test_features)
    return(features)


def extract_color_from_segmented_image(image):
    segments = segment_image(image, 3, 3)
    features = []
    h, s, v = [], [], []
    for seg in segments:
        hist, edges = make_hue_histogram(seg, c=2)
        idx = get_top_N_values_in_hist(hist, 1)[1]
        sat = np.mean(seg[:, :, 1])
        val = np.mean(seg[:, :, 2])
        idx = edges[idx]
        h.append(idx)
        s.append(sat)
        v.append(val)
    h = np.std(h)
    s = np.std(s)
    v = np.std(v)
    features.extend((h, s, v))
    return(features)
