'''
The algorithm which convert a noisy parallelogram
into a set of four points - an ideal parallelogram.
'''

import math
import itertools

import numpy as np
from matplotlib import pyplot
from shapely.geometry.polygon import LinearRing
from shapely.geometry import Point, Polygon


# Parallelogram detecting thresholds
LENGTH_T = 0.5
PERIMETER_T = 0.1
START_PEAK_HEIGHT_T = 1
BAD_HEIGHT = 2

# Hough accumulator resolution
THETA_RES = 1.0 / 1
RHO_RES = 1.0 / 1

# Peak height threshold decrement
PEAK_DEC = START_PEAK_HEIGHT_T / 10.0

# Maximum acceptable deviataion from original figure
MAX_DIV = 5

# Choose figure to run
FILE_PATH = "Testing_Figures/1.txt"


def get_figure(file_path):
    """
        Reads a set of points from the file.
    """
    assert isinstance(file_path, str), "file_path must be string."

    file = open(file_path, "r")
    points = []
    for line in file:
        row = line.split()
        points.append([float(row[0]), float(row[1])])

    assert len(np.shape(points)) == 2 and np.shape(points)[1] == 2, \
           "Set of points must be given as 2*n shaped array."
    return np.array(points)


def gen_shape_dict(shape):
    """
        Returns a dictionary containing shape charecteristics.

    Parameters
    ----------
    shape : ndarray
        Set of shape's points
        2D array containing data with 'float' type.

    Returns
    -------
    img : dict
        Dictionary containing image shape charecteristics

        Dict Entities
        -------------
            "points" : ndarray, 2D array containing data with 'float' type.
                Function argument "shape"
            "x_min" : float
                Minimal x-value in given set of points.
            "x_max" : float
                Maximal x-value in given set of points.
            "y_min" : float
                Minimal y-value in given set of points.
            "y_max" : float
                Maximal x-value in given set of points.
            "perimeter" : float
                Perimeter of given shape.
            "height" : float
                Vertical dimesion length of given shape.
            "width" : float
                Horisontal dimension length of given shape


    """

    img = {"points": shape,
           "x_min": np.ceil(np.amin(shape[:, 0])),
           "x_max": np.ceil(np.amax(shape[:, 0])),
           "y_min": np.ceil(np.amin(shape[:, 1])),
           "y_max": np.ceil(np.amax(shape[:, 1])),
           "perimeter": Polygon(shape).length}
    img["height"] = img["y_max"] - img["y_min"]
    img["width"] = img["x_max"] - img["x_min"]
    return img


def create_rho_theta(hough_dict, img):
    """
        Helper function for Hough Transform. Creates rho and theta dimension
    for Hough accumulator and put in hough_acc dictionary.

    Adds new entities to hough_acc dict:
    -----------------------------------
        "d_theta" : float
            Discrete step for theta space building.
        "d_rho" : float
            Discrete step for rho space building.
        "rho space" : ndarray, 1D array containing data with float type.
            Rho values in some range (depends on image dimensions) with
        descrete step "d_rho".
        "theta space" : ndarray, 1D array containing data with float type.
            Theta values in some range (depends on image dimensions) with
            descrete step "d_theta".

    Parameters
    ----------
    hough_acc : dict
        Dictionary containing Hough transformation parameters.
        See also: hough_transform() docs.
    img : dict
        Dictionary containing shape charecteristics.

    Returns
    -------
    None.

    """
    # Get image "shape"
    x_max = img["x_max"]
    y_max = img["y_max"]
    n_max = max(img["height"], img["width"])

    # Theta space
    hough_dict["d_theta"] = math.pi / (2*(n_max - 1)) * THETA_RES
    hough_dict["theta space"] = np.arange(-math.pi / 2, math.pi / 2,
                                          hough_dict["d_theta"])

    # Rho space
    distance = np.sqrt((x_max) ** 2 + (y_max) ** 2)
    hough_dict["d_rho"] = math.pi / 4 * RHO_RES
    hough_dict["rho space"] = np.arange(-distance, distance,
                                        hough_dict["d_rho"])


def fill_hs_acc(hough_acc, points):
    """
        Compute voices for every (rho, theta) pair in Hough Accumulator and
    writes it down in hough_acc["accumulator"].

    Parameters
    ----------
    hough_acc : dict
        Dictionary containing Hough transformation parameters.
        See also: hough_transform() docs.
    points : ndarray, 2D array containing data with 'float' type.
        Set of image's edge points

    Returns
    -------
    None.

    """
    # Get theta and rho space as variablese for reading convenience.
    theta_s = hough_acc["theta space"]
    rho_s = hough_acc["rho space"]

    # Iterate over each point in set
    for x, y in points:
        # Iterate over theta dimension in accumulator.
        # Accumulator Width == Theta Space length.
        for theta_idx in range(len(theta_s)):
            # Calculate rho value using equation of line in normal form
            rho_val = (x*math.cos(theta_s[theta_idx]) +
                       y*math.sin(theta_s[theta_idx]))

            # Find the rho spaces's element with the minimal difference to
            # calculated rho value. Index of this element is the collumn
            # index of accumulator cell to voice to.
            rho_idx = np.nonzero(np.abs(rho_s - rho_val) ==
                                 np.min(np.abs(rho_s - rho_val)))[0]
            hough_acc["accumulator"][rho_idx, theta_idx] += 1


def hough_transform(img):
    """
        Generates a new hough accumulator dictionary containing calculated
    accumulator and it's important charactesrstics

    Parameters
    ----------
    img : dict
        Dictionary containing image shape charecteristics.
        See also:  gen_shape_dict() docs

    Returns
    -------
    hough_acc : dict
        Dictionary containing Hough transformation parameters.

        Dict Entities
        -------------
            "d_theta" : float
                Discrete step for theta space building.
            "d_rho" : float
                Discrete step for rho space building.
            "rho space" : ndarray, 1D array containing data with float type.
                Rho values in some range (depends on image dimensions) with
            descrete step "d_rho".
            "theta space" : ndarray, 1D array containing data with float type.
                Theta values in some range (depends on image dimensions) with
                descrete step "d_theta".
            "accumulator" : ndarray, 2D array containing data with 'int' type
                Two-dimensional array, to detect the existence of a line on
                given image.
    """
    # Initialize an empty hough_acc dictionary
    hough_dict = {}

    # Add rho and theta accumulator dimensions in dictionary
    create_rho_theta(hough_dict, img)

    # Create an empty hough accumulator with (rho, theta) dimesions
    hough_dict["accumulator"] = np.zeros((len(hough_dict["rho space"]),
                                         len(hough_dict["theta space"])),
                                         dtype=int)

    # Count voices over the edge points
    fill_hs_acc(hough_dict, img["points"])

    return hough_dict


def find_peaks(hough_acc, img, peak_hieght_t):
    """
        Finds peaks in accumulator greater or equal to peak height threshold
    and writes it down in hough_acc dictionary.

        Adds new entities to hough_acc dict:
        ------------------------------------
            "Hough peaks" : list
                A list of peaks dictionaries containing peaks parameters:
                ---------------------------------------------------------
                "rho" : float
                    A rho value of found peak.
                "theta" : float
                    A theta value of found peak.
                "acc value" : int
                    An accumulator value (height) of found peak.
                "idx" : ndarray, 1D array of data with 'int' type
                    Row, collumn index of peak in accumulator.

    Parameters
    ----------
    hough_acc : dict
        Dictionary containing Hough transformation parameters.
        See also: hough_transform() docs.
    img : dict
        Dictionary containing image shape charecteristics.
        See also:  gen_shape_dict() docs
    peak_hieght_t : float
        Accumulator threshold parameter. Only those lines are returned that
        get enough votes ( >peak_height_t*"max accumulator value")

    Returns
    -------
    None.

    """
    parameters_list = []
    accumulator = hough_acc["accumulator"]
    max_acc_value = np.amax(accumulator)

    # Get indexes of all peaks that height is greater or equal then the highest
    # peak times peak height threshold
    min_height = max_acc_value * peak_hieght_t
    assert min_height > BAD_HEIGHT, \
        "Minimal peak height became to low. No parallelagrams were found."
    peak_idx_list = np.argwhere(accumulator >= min_height)

    # Add each peak parameters to parameters_list as dictionary
    for p_idx in peak_idx_list:
        acc_value = hough_acc["accumulator"][p_idx[0], p_idx[1]]
        rho = hough_acc["rho space"][p_idx[0]]
        theta = hough_acc["theta space"][p_idx[1]]
        parameters_list.append({"rho": rho,
                                "theta": theta,
                                "acc value": acc_value,
                                "idx": p_idx})

    hough_acc["Hough peaks"] = parameters_list


def get_paired_peaks(hough_acc, img):
    """
        Returns list of peaks pairs representing paralell lines.

    Parameters
    ----------
    hough_acc : dict
        A dictionary containing Hough transformation parameters.
        See also: hough_transform() docs.
    img : dict
        A dictionary containing image shape charecteristics.
        See also:  gen_shape_dict() docs.
    length_t : float
        A normolized treshold that verifies if line segment corresponding to
        peaks have aproximately the same length.

    Returns
    -------
    extended_peaks : list
        A list containing peak pairs parameters representing parallel lines.

    """
    extended_peaks = []
    # Theta threshold. Depends on theta space resolution
    theta_t = hough_acc["d_theta"] * 3

    # Iterate over all possible peaks pairs
    for peak1, peak2 in itertools.combinations(hough_acc["Hough peaks"], 2):
        theta1, theta2 = [peak1["theta"], peak2["theta"]]
        rho1, rho2 = [peak1["rho"], peak2["rho"]]
        acc_value1, acc_value2 = [peak1["acc value"], peak2["acc value"]]

        # Coorientation of sides and heights similarity conditions
        are_parallel = abs(theta1 - theta2) < theta_t
        are_simmilar_lengths = abs(acc_value1 - acc_value2) < \
                                  (LENGTH_T * (acc_value1 + acc_value2) * 0.5)
        if are_parallel and are_simmilar_lengths:
            # Generate new extended peak
            new_peak_dict = {"ksi1": rho1,
                             "ksi2": rho2,
                             "beta": 0.5 * (theta1 + theta2),
                             "acc value": 0.5 * (acc_value1 +
                                                 acc_value2),
                             "idx1": peak1["idx"],
                             "idx2": peak2["idx"]}
            extended_peaks.append(new_peak_dict)

    return extended_peaks


def gen_parallelograms_sides(peaks_list, hough_acc):
    """
        Returns a list of dictionaries containing parameters of 2 pairs of
    parallel peaks representing 4 sides of parallelogramms.

    Parameters
    ----------
    peaks_list : list
        A list of peak pairs parameters representing parallel lines.
    hough_acc : dict
        A dictionary containing Hough transformation parameters.
        See also: hough_transform() docs.

    Returns
    -------
    parallelograms_sides : list
        A list of dictionaries contining parameters of potential paralelograms
        sides.

    """
    parallelograms_sides = []

    # Iterate over all pairs of extended peaks
    for peak1, peak2 in itertools.combinations(peaks_list, 2):

        # Check if sides are adjecent (non zero angle).
        ang_dif = abs(peak1["beta"] - peak2["beta"])
        if ang_dif != 0:
            temp_dict = {"sides_a": peak1,
                         "sides_b": peak2,
                         "ang_dif": ang_dif}
            parallelograms_sides.append(temp_dict)

    return parallelograms_sides


def gen_expected_perimeters(valid_peaks_pairs):
    """
        Compute an expected perimeter for all parallelogramms in
    parallelogrammslist and writes it down in paraleloramms dictionary as:
        "exp per" : float
            An expected perimeter of parallelogramm.

    Parameters
    ----------
    valid_peaks_pairs : list
        A list of dictionaries contining parameters of potential paralelograms
        sides.

    Returns
    -------
    None.

    """
    for pair in valid_peaks_pairs:
        len_a = (abs(pair["sides_a"]["ksi1"] - pair["sides_a"]["ksi2"]) /
                 math.sin(pair["ang_dif"]))
        len_b = (abs(pair["sides_b"]["ksi1"] - pair["sides_b"]["ksi2"]) /
                 math.sin(pair["ang_dif"]))
        pair["exp_per"] = 2 * (len_a + len_b)


def validate_perimeter(valid_sides_pairs, actual_perimeter):
    """
        Returns a list of peaks representing parallelogramms with similar to
    actual shape perimeter.

    Parameters
    ----------
    valid_sides_pairs : list
        A list of dictionaries contining parameters of potential paralelograms
        sides.
    actual_perimeter : float
        Actual perimeter of the shape, that parallelogramms compares with.

    Returns
    -------
    valid_paralls : list
        List of parallelograms peaks which passed perimeter validation step.

    """
    valid_paralls = []
    for pair in valid_sides_pairs:
        exp_per = pair["exp_per"]

        # Check if current parallelogramm and image shape are aproximately the
        # same.
        val_par_condition = abs(actual_perimeter - exp_per) < \
                            PERIMETER_T * exp_per
        if val_par_condition:
            valid_paralls.append(pair)

    return valid_paralls


def get_sides_parameters(sides_list):
    """
        Return sides rho and theta parameters for all 4 lines in sides_list.

    Parameters
    ----------
    sides_list : list
       A list of peaks parameters corresponding to 4 lines.

    Returns
    -------
    list
        A list of rho and theta arameters for 4 lines.

    """
    beta_a = sides_list["sides_a"]["beta"]
    beta_b = sides_list["sides_b"]["beta"]

    side1 = [sides_list["sides_a"]["ksi1"], beta_a]
    side2 = [sides_list["sides_a"]["ksi2"], beta_a]
    side3 = [sides_list["sides_b"]["ksi1"], beta_b]
    side4 = [sides_list["sides_b"]["ksi2"], beta_b]

    return [side1, side2, side3, side4]


def get_best_shape(edge_points, poly_list):
    """
        Return a polygon with the smallest deviation of polygons in
    rings_list from image edge points.

    Parameters
    ----------
    points_arr : ndarray
        DESCRIPTION.
    rings_list : TYPE
        DESCRIPTION.

    Returns
    -------
    best_shape : TYPE
        DESCRIPTION.
    min_dist_sum : TYPE
        DESCRIPTION.

    """
    min_dist_sum = float('inf')  # Minimal sum of square distance from each edge point to polygon sides
    best_shape = None
    copy_points = np.copy(edge_points)

    # Find the polygon with the smallest deviation
    for ring in poly_list:
        dist_sum = 0
        shape = Polygon(ring)
        for point in copy_points:
            point = Point(point)

            # Compute a distance between point and polygon
            dist_sum += shape.exterior.distance(point) ** 2

        # Assign a new minimal distance to compare and best shape
        if dist_sum < min_dist_sum:
            min_dist_sum = dist_sum
            best_shape = ring

    return best_shape, min_dist_sum


def find_intersection(line1, line2):
    """
    Returns a tuple of lines intersection coordinates.

    Parameters
    ----------
    line1 : list
        Rho and theta parameters of 1st line.
    line2 : list
        Rho and theta parameters of 2nd line.

    Returns
    -------
    x_y : tuple
        A coordinates of lines intersection in xy coordinates.

    """
    rho1, theta1 = line1
    rho2, theta2 = line2

    # Solving a system of linear algebraic equations.
    a_matrix = np.array([[math.cos(theta1), math.sin(theta1)],
                         [math.cos(theta2), math.sin(theta2)]])
    b_matrix = np.array([rho1, rho2])
    x_y = np.linalg.solve(a_matrix, b_matrix)

    return x_y


def get_vertices(parall_sides):
    """
        Returns a list of vertices parallelogramm sides.

    Parameters
    ----------
    parall_sides : list
        A list of parallelogramms sides rho and theta parameters

    Returns
    -------
    list
        A list of paralelogramm vertices in ring form.

    """
    x1_y1 = find_intersection(parall_sides[0], parall_sides[3])
    x2_y2 = find_intersection(parall_sides[0], parall_sides[2])
    x3_y3 = find_intersection(parall_sides[1], parall_sides[2])
    x4_y4 = find_intersection(parall_sides[1], parall_sides[3])
    return [x1_y1, x2_y2, x3_y3, x4_y4]


def build_plot(polygon_vers, title):
    """
        Builds a plot of given ring shape
    """
    ring1 = LinearRing(polygon_vers)
    x, y = ring1.xy

    fig = pyplot.figure(1, figsize=(5, 5), dpi=90)
    polygon = fig.add_subplot(111)
    polygon.plot(x, y, marker='o')
    polygon.set_title(title)


def detect_paralls(hough_acc, image, peak_hieght_t):
    """
    Tries to detect parallelogramm on the given image. Returns it's vertices if
    it was found or trow an assert error otherwise.

    Parameters
    ----------
    hough_acc : dict
        A dictionary containing Hough transformation parameters.
        See also: hough_transform() docs.
    image : dict
        A dictionary containing image shape charecteristics.
        See also:  gen_shape_dict() docs.
    peak_hieght_t : float
        Accumulator threshold parameter. Uses in peaks (lines) finding.
        Only those lines are returned that get enough votes
        ( >peak_height_t*"max accumulator value").

    Returns
    -------
    vertices : list
        A list of vertices of detected parallelogramm.
    diff : float
        A sum of squared distances between detected parallelogramm and points
        from given set.

    """    
    # Find the best peaks from Hough accumulator 
    find_peaks(hough_acc, image, peak_hieght_t)

    # Find a peaks pairs representing parallel lines on image
    parallel_peaks = get_paired_peaks(hough_acc, image)

    # Parallelogramm has 4 sides, so we need at least 2 pairs of parallel lines.
    # If this condition doesn't meet repeat detection using lower peaks threshold.
    if len(parallel_peaks) < 2:
        return detect_paralls(hough_acc, image, peak_hieght_t - PEAK_DEC)

    # Find peaks quads representing to 4 sides of parallelograms
    paralls_peaks = gen_parallelograms_sides(parallel_peaks, hough_acc)

    # At least 1 parallelagramm must be found.
    # Otherwise repeat detection using lower peaks thresholds.
    if len(paralls_peaks) < 1:
        return detect_paralls(hough_acc, image, peak_hieght_t - PEAK_DEC)

    # Find a parallelogramms with perimeter similar
    # to a perimeter of given shape.
    gen_expected_perimeters(paralls_peaks)
    best_paralls = validate_perimeter(paralls_peaks, image["perimeter"])

    # At least 1 similar parallelogramm must be found.
    # Otherwise repeat a detection using lower peaks thresholds.
    if best_paralls == []:
        return detect_paralls(hough_acc, image, peak_hieght_t - PEAK_DEC)

    # Get a parallelogramms vertices finding intersection of its lines.
    sides_params = [get_sides_parameters(parall) for parall in best_paralls]
    paralls_verts = [get_vertices(sides) for sides in sides_params]

    # Find a parallelogramm with the smallest deviation from original shape.
    vertices, deviation = get_best_shape(image["points"], paralls_verts)

    # If no valid paralelograms were found repeat a detection using lower peaks
    # thresholds.
    if vertices is None:
        return detect_paralls(hough_acc, image, peak_hieght_t - PEAK_DEC)

    return vertices, deviation


def run_algorithm(points):
    """
    Runs a parallelogramm detection algorithm.    

    Parameters
    ----------
    points : ndarray
        Set of shape's points
        2D array containing data with 'float' type.

    Returns
    -------
    None.

    """

    # Get an image parameters for further comptetion
    image = gen_shape_dict(points)

    build_plot(image["points"], FILE_PATH)

    # Hough transform step
    peak_height_t = START_PEAK_HEIGHT_T
    hough_acc = hough_transform(image)

    # Parallelogramm detection step
    vertices, deviation = detect_paralls(hough_acc, image, peak_height_t)

    build_plot(vertices, "Diff: " + str(deviation))


run_algorithm(get_figure(FILE_PATH))
