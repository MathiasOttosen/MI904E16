import numpy as np
from math import sqrt, pow
from skimage import feature
from skimage.color import rgb2gray, rgb2hsv, gray2rgb
from skimage.transform import probabilistic_hough_line as phl
from skimage.measure import find_contours


src = 'experiments/experiment_lite/'


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
    contours = find_contours(rgb2gray(image), 0.5)
    mean_straight_length = compute_avg_straght_length(lines)
    mean_contour_length = compute_avg_contour_length(contours)
    curvature = avg_diff_straight_contour(contours)
    dist_lines = distribution_of_lines(image)
    line_features = [mean_straight_length, mean_contour_length,
                     curvature]
    line_features.extend(dist_lines)
    return(line_features)


def distribution_of_lines(image, width=3, height=3):
    segments = segment_image(image, width, height)
    num_lines = []
    for s in segments:
        contours = find_contours(rgb2gray(s), 0.5)
        num_lines.append(len(contours))
    if np.sum(num_lines) == 0:
        return(num_lines)
    else:
        dist = [float(l) / np.sum(num_lines) for l in num_lines]
        return(dist)


def avg_diff_straight_contour(contours):
    def diff_straight_contour(contour):
        c_length = contour_length(contour)
        s_length = line_length(contour[-1], contour[0])
        return(abs(s_length - c_length))
    diffs = [diff_straight_contour(c) for c in contours]
    return(np.mean(diffs))


def segment_image(image, h_segs, v_segs):
    height = image.shape[0] / h_segs
    width = image.shape[1] / v_segs
    segments = []
    for i in range(1, h_segs+1):
        for n in range(1, v_segs+1):
            seg = image[height * (i-1):height * i, width * (n-1):width * n]
            segments.append(seg)
    return(segments)


def contour_length(contour):
    arc = line_length(contour[1], contour[0])
    for n in range(2, len(contour)):
        arc = arc + line_length(contour[n], contour[n-1])
    return(arc)


def line_length(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    l = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    return(l)


def compute_avg_straght_length(lines):
    lengths = [line_length(l[0], l[1]) for l in lines]
    return(np.mean(lengths))


def compute_avg_contour_length(contours):
    lengths = [contour_length(c) for c in contours]
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
        if t == 1 and n != 1:
            index.extend(idx)
            r1, r2 = np.random.randint(0, high=len(hist), size=2)
            while r1 == index[0] or r2 == index[0]:
                r1, r2 = np.random.randint(0, high=len(hist), size=2)
            index.extend((r1, r2))
            break
        if len(idx) != 1:
            i += len(idx) - 1
        index.extend(idx)
        tmp_hist = np.delete(tmp_hist, idx)
        i += 1
    top.extend(hist[index])
    return(np.sort(top), np.sort(index))


def quick_test(images, test_images=None):
    features = []
    for i in images:
        try:
            hsv_img = rgb2hsv(i)
        except ValueError:
            rgb_img = gray2rgb(i)
            hsv_img = rgb2hsv(rgb_img)
        hist = make_hue_histogram(hsv_img)[0]
        sat = np.mean(hsv_img[:, :, 1])
        val = np.mean(hsv_img[:, :, 2])
        f = [sat, val]
        f.extend(hist)
        features.append(f)
    features = np.vstack(features)
    if test_images is not None:
        test_features = quick_test(test_images)
        return(features, test_features)
    return(features)


def color_std(image):
    segments = segment_image(image, 3, 3)
    c = []
    for s in segments:
        tmp_hist, edges = make_hue_histogram(s)
        idx = get_top_N_values_in_hist(tmp_hist, 1)[1]
        c.append(edges[idx])
    std = np.std(c)
    return(std)
