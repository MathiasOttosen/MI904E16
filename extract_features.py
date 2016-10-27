import numpy as np
from skimage import feature
from skimage.color import rgb2gray


# The main extraction method.
# The extract_func is a tuple containing the extraction method and
# optional parameters
def extract_features(images, extract_func, test_images=None):
    features = []
    # Most of these if statements are for special cases, such as the peaks method
    # that require the indices parameter set to false, see extract_peaks
    if len(extract_func) == 1:  # If the length of the tuple is one, there are no optional parameters
        if extract_func[0] is feature.corner_peaks:
            features = extract_peaks(images, feature.corner_peaks)  # We use the same method for both peak methods
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
        i = extract_func(i).flatten()
        features.append(i)
    return(features)


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


# This is for using the peak methods, cornre_peak and peak_local_max
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
        i = ibp(i, 24, 8).flatten()
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
                                        [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                        prop=prop)).flatten()
        greycoprops.append(i)
    return(greycoprops)
