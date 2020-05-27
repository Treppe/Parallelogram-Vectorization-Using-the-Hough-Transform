'''
The algorithm which convert a noisy parallelogram
into a set of four points - an ideal parallelogram.
'''

import math
import itertools
from copy import deepcopy

import numpy as np
from matplotlib import pyplot
from shapely.geometry.polygon import LinearRing
from shapely.geometry import Point, Polygon


# Parallelogram detecting thresholds
LENGHT_T = 0.8                    # Used in get_paired_peaks() for coorientation validation step
PERIMETER_T = 0.1                 # Used in validate_perimeter() for perimeter validation step


THETA_RES = 1.0 / 2
RHO_RES = 1.0 / 4
START_PEAK_HEIGHT_T = 1
PEAK_DEC = 0.1
MAX_PEAKS_PAIRS = float("inf")               # Proportional to runtime

MAX_DIV = 5

# Choose figure to run
FILE_PATH = "Testing_Figures/1.txt"


def get_figure(file_path):
    """
    Reads set of points from the file.
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
        Dictionary containing shape charecteristics
        KEYS DESCRIPTION:
            points: ndarray
                Function argument "shape"
                2D array containing data with 'float' type.
            "x_min" : float
                Minimal x-value in given set of points
            "x_max" : float
                Maximal x-value in given set of points
            "y_min" : float
                Minimal y-value in given set of points
            "y_max" : float
                Maximal x-value in given set of points
            "perimeter" : float
                Perimeter of given shape
            "height" : float
                Vertical dimesion length of given shape
            "width" : float
                Horisontal dimension length of given shape
                

    """
            
    img = {"points": shape,
            "x_min": np.ceil(np.amin(shape[:, 0])),
            "x_max": np.ceil(np.amax(shape[:, 0])),
            "y_min": np.ceil(np.amin(shape[:, 1])),
            "y_max": np.ceil(np.amax(shape[:, 1])),
            "perimeter": Polygon(shape).length}       # 
    img["height"] = img["y_max"] - img["y_min"]
    img["width"] = img["x_max"] - img["x_min"]
    return img


def create_rho_theta(hough_acc, img):
    """
    Helper function for Hough Transform. Creates rho and theta dimension for 
    Hough accumulator and put in hough_acc dictionary. 
    
    Add new keys to hough_acc dict:
        "d_rho"

    Parameters
    ----------
    hough_acc : dict
        Dictionary containing Hough transformation parameters.
        See also: hough_transform()
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
    hough_acc["d_theta"] = math.pi / (2*(n_max - 1)) * THETA_RES
    hough_acc["theta space"] = np.arange(-math.pi / 2, math.pi / 2, 
                                         hough_acc["d_theta"])
    
    # Rho space
    distance = np.sqrt((x_max) ** 2 + (y_max) ** 2)
    hough_acc["d_rho"] = math.pi / 4 * RHO_RES
    hough_acc["rho space"] = np.arange(-distance, distance, hough_acc["d_rho"])
    

def fill_hs_acc(hough_acc, points):
    """
    Compute voices for every (rho, theta) pair in Hough Accumulator and writes
    it down in hough_acc dictionary.
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
    """
    hough_acc = {}
    create_rho_theta(hough_acc, img)
    
    # Create an empty hough space with (rho, theta) dimesions
    hough_acc["accumulator"] = np.zeros((len(hough_acc["rho space"]), 
                                         len(hough_acc["theta space"])))
    
    fill_hs_acc(hough_acc, img["points"])
    return hough_acc


def find_peaks(hough_acc, img, peak_hieght_t):
    parameters_list = []
    accumulator = hough_acc["accumulator"]
    max_acc_value = np.amax(accumulator)
    peak_idx_list = np.argwhere(accumulator >= max_acc_value * peak_hieght_t)
    
    for p_idx in peak_idx_list:
        acc_value = hough_acc["accumulator"][p_idx[0], p_idx[1]]
        rho = hough_acc["rho space"][p_idx[0]]
        theta = hough_acc["theta space"][p_idx[1]]
        parameters_list.append({"rho": rho,
                                "theta": theta,
                                "acc value": acc_value,
                                "idx": p_idx})
    
    hough_acc["Hough peaks"] = parameters_list


def rucursive_call(hough_acc, img, counter, peak_hieght_t):
    copy_acc = deepcopy(hough_acc) 
    
    # Redefine MIN PEAK HEIGHT threshold 
    peak_hieght_t -= 0.1
    assert peak_hieght_t != START_PEAK_HEIGHT_T, "premax_val the same as max_val"
    
    # Repeat peaks search with new FACTOR
    copy_acc["Hough peaks"] = []
    find_peaks(copy_acc, img, peak_hieght_t)
    return copy_acc, peak_hieght_t
    

def get_paired_peaks(hough_acc, img):
    """

    Parameters
    ----------
    ht_acc : np.array
        Hough transform accumulator matrix C (rho by theta)
    peaks : np.array
        Rho and theta pairs representing potential peaks
    rhos : list
        Ordered array of rhos represented by rows in C
    thetas : list
        Ordered array of thetas represented by columns in C
    theta_T : float
        Angular threshold that determines if peaks correspond to a parallel
        line
    len_T : float, optional
        Normolized threshold that verifies if line segment corresponding
        to pair of peaks have approximately the same length.
        The default is 0.5.

    Returns
    -------
    satisfying_pairs : list
        Pairs of peaks occuring at the same orientation theta, and with similar
        heights.

    """
    extended_peaks = []
    theta_t = hough_acc["d_theta"] * 3                                              # Theta threshold. Depends on theta space resolution                                     
    
    # Pair matching peaks. Try to get parallel lines with of aprox. the same "length"
    for peak1, peak2 in itertools.combinations(hough_acc["Hough peaks"], 2):
        theta1, theta2 = [peak1["theta"], peak2["theta"]]
        rho1, rho2 = [peak1["rho"], peak2["rho"]]
        acc_value1, acc_value2 = [peak1["acc value"], peak2["acc value"]] 
        
        # Coorientation of sides conditions
        are_parallel = abs(theta1 - theta2) < theta_t
        is_apropriate_lenght = abs(acc_value1- acc_value2) < LENGHT_T * (acc_value1 + acc_value2) * 0.5
        if are_parallel and is_apropriate_lenght:
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


def gen_parallelograms_sides(peaks_list, hough_acc, img):
    parallelograms_sides = []
    
    for peak1, peak2 in itertools.combinations(peaks_list, 2):
        ang_dif = abs(peak1["beta"] - peak2["beta"])
        if ang_dif != 0:
            temp_dict = {"sides_a" : peak1,
                         "sides_b" : peak2,
                         "ang_dif" : ang_dif}
            parallelograms_sides.append(temp_dict)

    return parallelograms_sides


def gen_expected_perimeters(valid_peaks_pairs):
    for pair in valid_peaks_pairs:
        len_a = (abs(pair["sides_a"]["ksi1"] - pair["sides_a"]["ksi2"]) / 
                 math.sin(pair["ang_dif"]))
        len_b = (abs(pair["sides_b"]["ksi1"] - pair["sides_b"]["ksi2"]) / 
                 math.sin(pair["ang_dif"]))
        pair["exp_per"] = 2 * (len_a + len_b)


def validate_perimeter(valid_sides_pairs, actual_perimeter):
    valid_paralls = []
    for pair in valid_sides_pairs:
        exp_per = pair["exp_per"]
        val_par_condition = (abs(actual_perimeter - exp_per) <
                              PERIMETER_T * exp_per)
        if val_par_condition:
            valid_paralls.append(pair)
                
    return valid_paralls


def get_sides_parameters(sides_pair):
    beta_a = sides_pair["sides_a"]["beta"]
    beta_b = sides_pair["sides_b"]["beta"]
    
    side1 = [sides_pair["sides_a"]["ksi1"], beta_a]
    side2 = [sides_pair["sides_a"]["ksi2"], beta_a]
    side3 = [sides_pair["sides_b"]["ksi1"], beta_b]
    side4 = [sides_pair["sides_b"]["ksi2"], beta_b]
    
    return [side1, side2, side3, side4]
    
    
def get_best_shape(points_arr, rings_list):
    min_dist_sum = float('inf')
    best_shape = None
    copy_points = np.copy(points_arr)
    ###
    test_list = []
    for ring in rings_list:
        dist_sum = 0
        shape = Polygon(ring)
        too_far = False
        for point in copy_points:
            point = Point(point)
            dist_sum += shape.exterior.distance(point) ** 2
            if dist_sum > MAX_DIV:
                too_far = True
                break
        test_list.append((ring, dist_sum))
        if dist_sum < min_dist_sum and not too_far:
            min_dist_sum = dist_sum
            best_shape = ring
            
    return best_shape, min_dist_sum


def find_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    a_matrix = np.array([[math.cos(theta1), math.sin(theta1)],
                         [math.cos(theta2), math.sin(theta2)]])
    b_matrix = np.array([rho1, rho2])
    x_y = np.linalg.solve(a_matrix, b_matrix)
    return x_y


def get_vertices(parall_sides):
    x1_y1 = find_intersection(parall_sides[0], parall_sides[3])
    x2_y2 = find_intersection(parall_sides[0], parall_sides[2])
    x3_y3 = find_intersection(parall_sides[1], parall_sides[2])
    x4_y4 = find_intersection(parall_sides[1], parall_sides[3])
    return [x1_y1, x2_y2, x3_y3, x4_y4]


def build_plot(polygon_vers, title):
    ring1 = LinearRing(polygon_vers)
    x, y = ring1.xy
     
    fig = pyplot.figure(1, figsize=(5, 5), dpi=90)
    polygon = fig.add_subplot(111)
    polygon.plot(x, y, marker='o')
    polygon.set_title(title)
    
    

def detect_paralls(hough_acc, image, peak_hieght_t):
    assert peak_hieght_t > 0.3, "No parallelograms were found"
    print(peak_hieght_t)
    
    find_peaks(hough_acc, image, peak_hieght_t)
    paired_peaks = get_paired_peaks(hough_acc, image)
    if len(paired_peaks) < 2:
        return detect_paralls(hough_acc, image, peak_hieght_t - PEAK_DEC)
    assert len(paired_peaks) < MAX_PEAKS_PAIRS, "No parallelograms were found"
    
    potent_paralls = gen_parallelograms_sides(paired_peaks, hough_acc, 
                                              image)                            # Parameters of sets of 4 sides
                                                                                # potential desired parallelogram candidates
    if len(potent_paralls) < 1:
        return detect_paralls(hough_acc, image, peak_hieght_t - PEAK_DEC)
        
        
    gen_expected_perimeters(potent_paralls)
    best_paralls = validate_perimeter(potent_paralls, image["perimeter"])                        
    if best_paralls == []:
        return detect_paralls(hough_acc, image, peak_hieght_t - PEAK_DEC)
    
    sides_params = [get_sides_parameters(parall) for parall in best_paralls]
    paralls_verts = [get_vertices(sides) for sides in sides_params]
    vertices, deviation = get_best_shape(image["points"], paralls_verts)
    if vertices == None:
        return detect_paralls(hough_acc, image, peak_hieght_t - PEAK_DEC)
    
    return vertices, deviation
    

def run_algorithm(points):
    '''
    Parameters
    ----------
    figure : TYPE, string
        DESCRIPTION. Name of figure to detect
    enhanced : bool
        Determine if normal or encanced accumulator is required.
        The default is True

    Returns
    -------
    thetas : TYPE, np.array
        DESCRIPTION. Ordered array of thetas represented by columns in C
    rhos : TYPE, np.array
        DESCRIPTION. Ordered array of rhos represented by rows in C
    C : TYPE, np.array
        DESCRIPTION. Hough accumulator - array representing Hough Space and
        accumulating "voices" for each point in this space.
    C_ench : TYPE, np.array
        DESCRIPTION. Enchanced version of Hough Accumulator
    rho_theta_pairs : np.array
        DESCRIPTION. Ordered list of top rho and theta pairs
        (has the most value in Hough accumulator)
    x_y_pairs : np.array
        DESCRIPTION. x and y coordinates corresponding to sinusoids
        in Hough Space

    '''
    
    peak_hieght_t = START_PEAK_HEIGHT_T
    image = gen_shape_dict(points)
    build_plot(image["points"], FILE_PATH)
    hough_acc = hough_transform(image)
    
    vertices, deviation = detect_paralls(hough_acc, image, peak_hieght_t)
    
    build_plot(vertices, "Diff: " + str(deviation))
    
    

run_algorithm(get_figure(FILE_PATH))
