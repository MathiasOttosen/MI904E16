import numpy as np
from skimage import feature
from skimage.color import rgb2gray
from sympy import N, Segment
from sympy.core.numbers import Zero


# The main extraction method.
# The extract_func is a tuple containing the extraction method and
# optional parameters
def extract_features(images, extract_func, test_images=None):
    features = []
    # Most of these if statements are for special cases, such as the peaks method
    # that require the indices parameter set to false, see extract_peaks
    if len(extract_func) == 1:  # If the length of the tuple is one, there are no optional parameters
        if 'corner_' in extract_func[0].func_name:
            features = extract_corners(images, extract_func[0])
        elif extract_func[0] is feature.local_binary_pattern:
            features = extract_lbp(images, feature.local_binary_pattern)
        elif extract_func[0] is feature.peak_local_max:
            features = extract_peaks(images, feature.peak_local_max)
        else:
            features = general_feature_extractor(images, extract_func[0])
    # If there are more parameters in the tuple
    else:
        if extract_func[0] is feature.greycoprops:
            features = extract_greycoprops(images, extract_func[0], extract_func[1])
        elif extract_func[0] is feature.structure_tensor_eigvals:
            features = extract_structure_tensor_eigvals(images, extract_func[0], extract_func[1])
        else:
            print("I don't understand the method you have given me")
            return(-1)
    # When we have our features we stack them in a numpy.ndarray which will be used by the classifier
    features = np.vstack(features)
    # If there are test images, we call the method again using these and out put both sets of features
    if test_images is not None:
        test_features = extract_features(test_images, extract_func)
        return(features, test_features)
    return(features)


# The general feature extractor, for the extraction methods
# that do not need special parameters
def general_feature_extractor(images, extract_func):
    features = []
    for i in images:
        i = rgb2gray(i)
        i = extract_func(i).flatten()
        features.append(i)
    return(features)


def extract_corners(images, corner_func):
    corners = []
    for i in images:
        i = rgb2gray(i)
        corner = corner_func(i)
        corners.append(feature.corner_peaks(corner))
    return(corners)


def get_line_indices(edge_map):
    indices = []
    it = np.nditer(edge_map, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            indices.append(it.multi_index)
        it.iternext()
    return(indices)


def sort_after_second_coordinate(indices):
    l = list()
    for t in indices:
        l.append((t[1], t[0]))
    l.sort()
    res = list()
    for t in l:
        res.append((t[1], t[0]))
    return(res)


def find_shared_coordinates(line_seg, hlines, vlines):
    lines = []
    index = 0
    if type(line_seg.slope) is Zero:
        for s in hlines:
            if (s.p1 == line_seg.p1 or s.p1 == line_seg.p2) or \
               (s.p2 == line_seg.p1 or s.p2 == line_seg.p2):
                lines.append(s)
                index = hlines.index(s)
    else:
        for s in vlines:
            if (s.p1 == line_seg.p1 or s.p1 == line_seg.p2) or \
               (s.p2 == line_seg.p2 or s.p2 == line_seg.p1):
                lines.append(s)
                index = vlines.index(s)
    return(lines, index)


def remove_visited(visited, hlines, vlines):
    if len(visited) != 0:
        for s in visited:
            if s in hlines:
                hlines.remove(s)
            else:
                vlines.remove(s)
    return(hlines, vlines)


def get_end_point_horizontal(indices, steps, start):
    e_index = indices.index(start)
    end = start
    peek = True
    while peek:
        if (e_index + 1 != len(indices)) and \
           (end[0] == indices[e_index + 1][0]) and \
           (abs(end[1] - indices[e_index + 1][1]) <= steps):
            end = indices[e_index + 1]
            e_index += 1
        else:
            peek = False
    return(end, e_index)


def get_end_point_vertical(indices, steps, start):
    e_index = indices.index(start)
    end = start
    peek = True
    if e_index + 1 == len(indices):
        return(end, e_index)
    while peek:
        if (e_index + 1 != len(indices)) and \
           (end[1] == indices[e_index + 1][1]) and \
           (abs(end[0] - indices[e_index + 1][0]) <= steps):
            end = indices[e_index + 1]
            e_index += 1
        else:
            peek = False
    return(end, e_index)


def get_straight_lines(canny_image):
    print('Finding indices ...')
    indices = get_line_indices(canny_image)
    print('Done')
    lines = []
    print('Finding horizontal lines...')
    hlines = find_horizontal_lines(indices)
    print('Done')
    print('Finding vertical lines...')
    vlines = find_vertical_lines(indices)
    print('Done')
    l_hlines, l_vlines = len(hlines), len(vlines)
    print('Finding initial diagional lines')
    diag_lines, hlines, vlines = find_diagonal_lines(hlines, vlines)
    print('Done')
    while (len(hlines) > 0) and \
          (l_hlines != len(hlines) or l_vlines != len(vlines)):
        print('Finding next diagonal line')
        tmp_lines, hlines, vlines = find_diagonal_lines(hlines, vlines)
        diag_lines.extend(tmp_lines)
        l_hlines, l_vlines = len(hlines), len(vlines)
        print('Done')
    lines.extend(hlines)
    lines.extend(vlines)
    lines.extend(diag_lines)
    return(lines)


def find_horizontal_lines(indices, peek=3):
    hlines = []
    start, end = None, None
    n = 0
    while n < len(indices) - 1:
        start = indices[n]
        end, n = get_end_point_horizontal(indices, peek, start)
        if end is not start:
            hlines.append(Segment(start, end))
        elif n + 1 == len(indices):
            break
        else:
            n += 1
    return(hlines)


def find_vertical_lines(indices, peek=3):
    vlines = []
    start, end = None, None
    indices = sort_after_second_coordinate(indices)
    n = 0
    while n < len(indices) - 1:
        start = indices[n]
        end, n = get_end_point_vertical(indices, peek, start)
        if end is not start:
            vlines.append(Segment(start, end))
        elif n + 1 == len(indices):
            break
        else:
            n += 1
    return(vlines)


def find_diagonal_lines(hlines, vlines, first=None):
    if first is None:
        first = hlines[0]
    actual = [first]
    diag_lines = []
    forks = []
    h_index, v_index = 0, 0
    print('   Finding initial point')
    if type(first.slope) is not Zero:
        next_seg, v_index = find_shared_coordinates(first, hlines, vlines)
    else:
        next_seg, h_index = find_shared_coordinates(first, hlines, vlines)
    if len(next_seg) > 1:
        forks.extend(next_seg[1:])
        next_seg = next_seg[0]
        actual.append(next_seg)
        print('    Was multiple' + str(next_seg.p1))
    elif len(next_seg) == 1:
        next_seg = next_seg[0]
        actual.append(next_seg)
        print('    Was single' + str(next_seg.p1))
    else:
        print('    Was None')
        next_seg = None
    print('Done')
    while next_seg is not None:
        if type(next_seg.slope) is not Zero:
            next_seg, v_index = find_shared_coordinates(next_seg,
                                                        hlines,
                                                        vlines[v_index + 1:])
        else:
            next_seg, h_index = find_shared_coordinates(next_seg,
                                                        hlines[h_index + 1:],
                                                        vlines)
        if len(next_seg) > 1:
            forks.extend(next_seg[1:])
            next_seg = next_seg[0]
            actual.append(next_seg)
            print('   Found multiple line segs' + str(next_seg.p1))
        elif len(next_seg) == 1:
            next_seg = next_seg[0]
            actual.append(next_seg)
            print('    Found exactly one line seg' + str(next_seg.p1))
        else:
            print('    Found nothing')
            next_seg = None
    initial_line = Segment(actual[0].p1, actual[-1].p2)
    diag_lines.extend(RDP(actual, initial_line))
    hlines, vlines = remove_visited(actual, hlines, vlines)
    while len(forks) > 0:
        print('    Running through forks')
        diag_lines.extend(find_diagonal_lines(hlines, vlines, first=forks.pop(-1)))
    print('Done')
    return(diag_lines, hlines, vlines)


def RDP(actual, line, thresh=10):
    print('        Segmenting diagonal line')
    line_segs = []
    for s in actual:
        distance = max(N(line.distance(s.p1)), N(line.distance(s.p2)))
        furthest_point = s.p1 if distance == N(line.distance(s.p1)) else s.p2
        if distance > thresh:
            print('             Cut')
            index = actual.index(s)
            first_line = Segment(line.p1, furthest_point)
            second_line = Segment(furthest_point, line.p2)
            line_segs.extend(RDP(actual[:index], first_line, thresh=thresh))
            line_segs.extend(RDP(actual[index:], second_line, thresh=thresh))
        else:
            line_segs.append(line)
    print('Done')
    return(line_segs)


# Here we extract the structure tensor eigenvalues for the images
# The 'large' parameter is a boolean which is true if we want
# the large eigenvalues and false if we want the small ones
def extract_structure_tensor_eigvals(images, steig, large):
    eigenvalues = []
    for i in images:
        i = rgb2gray(i)
        Axx, Axy, Ayy = feature.structure_tensor(i)
        i = steig(Axx, Axy, Ayy)[0].flatten() if large else steig(Axx, Axy, Ayy)[1].flatten()
        eigenvalues.append(i)
    return(eigenvalues)


# This is for using the peak method, peak_local_max
def extract_peaks(images, peaks):
    corner_peaks = []
    for i in images:
        i = rgb2gray(i)
        i = peaks(i, indices=False).flatten()
        corner_peaks.append(i)
    return(corner_peaks)


# This extracts the local binary pattern in the image
# It requires an amount of neighbors and a radius to work,
# which here is 24 and 8, respectively
def extract_lbp(images, ibp):
    local_binary_patterns = []
    for i in images:
        i = rgb2gray(i)
        i = ibp(i, 24, 8)
        local_binary_patterns.append(i)
    return(local_binary_patterns)


# This extracts the grey co-occurance matrix properties from the image
# It is used for all six props: contrast, dissimilarity, homogeneity
#                               ASM, energy, and correlation
# The actual grey co-occurance matrix is instantiated with reference values
# from the scikit image website
def extract_greycoprops(images, greyco, prop):
    greycoprops = []
    for i in images:
        i = rgb2gray(i)
        i = greyco(feature.greycomatrix(i, [1],
                                        [0, np.pi/4, np.pi/2, 3*np.pi/4]),
                   prop=prop).flatten()
        greycoprops.append(i)
    return(greycoprops)
