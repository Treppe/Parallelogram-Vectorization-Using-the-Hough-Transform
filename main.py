'''
The algorithm which convert a noisy parallelogram
into a set of four points - an ideal parallelogram.
'''

import math
import itertools

import numpy as np
from matplotlib import pyplot
from shapely.geometry.polygon import LinearRing
from shapely.geometry import Polygon


# Parallelogram detecting thresholds
MIN_ACCEPT_HEIGHT = np.array(4)
LENGHT_T = 0.3
DIST_T = 0.5
PERIMETER_T = 0.01

# Rho and theta discretive step (accumulator's resolution)
RHO_RES = 0.1
#RHO_RES = 0.1 * MIN_ACCEPT_HEIGHT
# THETA_RES = 0.017453*10
THETA_RES = 0.01745331 # 0.0174533 rad = 1 grad
THETA_T = THETA_RES * 3
MIN_PEAKS_TO_FIND = 100

# MASK CONSTATS
MASK_HEIGHT = 1
MASK_WIDTH = 1

# Image Enhencement Constants
ENH_H = 5
ENH_W = 5

# Choose figure to run
FILE_PATH = "Testing_Figures/1.txt"


def assign_figure(file_path):
    file = open(file_path, "r")
    points = []
    for line in file:
        row = line.split()
        points.append([float(row[0]), float(row[1])])
    return np.array(points)


def find_min_max_points(points):
    """
    Find the highest value in each column of vector
    Parameters
    ----------
    point : np.array
        2D array of points where max values find to

    Returns
    -------
    x_max : float
        The highest value by x axis
    y_max : float
        The higest value by y axis

    """
    x_min = np.ceil(np.amin(points[:, 0]))
    x_max = np.ceil(np.amax(points[:, 0]))
    y_min = np.ceil(np.amin(points[:, 1]))
    y_max = np.ceil(np.amax(points[:, 1]))

    return x_min, x_max, y_min, y_max


def create_rho_theta(points):
    """
    Helper function for Hough Transform
    Creates rho and theta dimension for enchaced Hough accumulator

    Parameters
    ----------
    x_max : float
        The highest value by x axis
    y_max : float
        The highest value by y axis

    Returns
    -------
    theta : np.array
        Empty array representing theta dimension
    rho : np.array
        Empty array representing rho dimension

    """
    global THETA_T
    x_min, x_max, y_min, y_max = find_min_max_points(points)
    n_max = max(x_max - x_min, y_max - y_min)
    
    d_theta = math.pi / (2*(n_max - 1))
    THETA_T = 3 * d_theta
    d_rho = math.pi / 4
    
    theta = np.arange(-math.pi/2, math.pi/2, d_theta)
    
    distance = np.sqrt((x_max + 1)**2 + (y_max + 1)**2)
    rho = np.arange(-distance, distance, d_rho)
# =============================================================================
#     theta = np.linspace(-math.pi / 2, 0.0,
#                         math.ceil(math.pi / (2*THETA_RES) + 1))
#     theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
# 
#     distance = np.sqrt((x_max + 1)**2 + (y_max + 1)**2)
#     dummy = math.ceil(distance / RHO_RES)
#     nrho = 2*dummy + 1
#     rho = np.linspace(-dummy*RHO_RES, dummy*RHO_RES, nrho)
# =============================================================================
    
    return rho, theta


def fill_hs_acc(ht_acc, points, rho, theta):
    """
    Compute voices for every (rho, theta) pair in Hough Accumulator and update
    it values.

    Parameters
    ----------
    empty_acc : np.array
        Empty Hough accumulator
    points : np.array
        2D array of points to transform
    rho : float
        Ordered array of rhos represented by rows in C
    theta : np.array
        Ordered array of thetas represented by columns in C

    Returns
    -------
    ht_acc : np.array
        Filled Hough accumulator

    """
    
    for x__, y__ in points:
        for theta_idx in range(len(theta)):
            rho_val = x__*math.cos(theta[theta_idx]) + \
                    y__*math.sin(theta[theta_idx])
            rho_idx = (np.nonzero(np.abs(rho-rho_val) ==
                       np.min(np.abs(rho-rho_val)))[0])
            ht_acc[rho_idx, theta_idx] += 1
    ht_acc[ht_acc < MIN_ACCEPT_HEIGHT] = 0
    
    return ht_acc


def hough_transform(points):
    """
    Builds a Hough H(theta, rho) space for given arrai of (x,y) coordinates

    Parameters
    ----------
    points : np.array
        2D array of points to transform
    theta_res : float
        Theta resolution. Determines discretisation step for theta.
        The default is 1.0.
    rho_res : TYPE, float
        Rho resolution. Determines discretisation step for rho.
        The default is 1.0
    enhanced : bool
        Determine if normal or encanced accumulator is required.
        The default is True
    Returns
    -------
    theta : np.array
        Ordered array of thetas represented by columns in C
    rho : np.array
        Ordered array of rhos represented by rows in C
    C : np.array
        Hough accumulator - array representing Hough Space and
        accumulating "voices" for each point in this space.
    C_ench : np.array
        Enchanced version of Hough Accumulator
    theta_T : float
        A theta threshold required for further computation
    """
    
    
    rho, theta = create_rho_theta(points)
    ht_acc = np.zeros((len(rho), len(theta)))
    fill_hs_acc(ht_acc, points, rho, theta)
    print(np.shape(ht_acc))
    return rho, theta, ht_acc

def enhance(rect_region, heigth = 1):
    width = np.shape(rect_region)
    
    integr = np.sum(rect_region)
    region_enh = (heigth * width / integr) * rect_region ** 2
    
    return region_enh


def find_peaks_v2(ht_acc, rhos, thetas):
    rho_theta_acc = []
    
    for row in range(np.shape(ht_acc)[0]):
        if np.sum(ht_acc[row, :]) != 0:
            ht_acc_enh = enhance(ht_acc[row, :])
            col = np.argmax(ht_acc_enh)
            acc_value = ht_acc[row, col]
            if acc_value != 0:
                rho = rhos[row]
                theta = thetas[col]
                rho_theta_acc.append([rho, theta, acc_value])
    
    return rho_theta_acc

def find_peaks(ht_acc, rhos, thetas):
    rho_theta_acc = []
    
    while True:
        acc_max = np.amax(ht_acc)
        
        if acc_max >= MIN_ACCEPT_HEIGHT:
            # Get an index of the highest peak
            peak_idx_list = np.argwhere(ht_acc == acc_max)
            
            for peak_idx in peak_idx_list:
                acc_value = ht_acc[peak_idx[0], peak_idx[1]]
                if acc_value != 0:
                    rho = rhos[peak_idx[0]]
                    theta = thetas[peak_idx[1]]
                    mask_origin = np.array([(peak_idx[0] - MASK_HEIGHT // 2),
                                            (peak_idx[1] - MASK_WIDTH // 2)])
                    mask_origin[mask_origin < 0] = 0
    
                    ht_acc[mask_origin[0]: mask_origin[0] + MASK_HEIGHT,
                           mask_origin[1]: mask_origin[1] + MASK_WIDTH] = 0
                    rho_theta_acc.append((rho, theta, acc_value))
        else:
            break
        
    return list(rho_theta_acc)


def get_cooriented_pairs(peaks, rhos, thetas):
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
    for peak1, peak2 in itertools.combinations(peaks, 2):
        rho1, theta1, acc_value1 = peak1
        rho2, theta2, acc_value2 = peak2
        
        is_parallel = abs(theta1 - theta2) < THETA_T
        is_apropriate_lenght = (abs(acc_value1 - acc_value2) <
                                LENGHT_T * (acc_value1 + acc_value2) * 0.5)
        
        if is_parallel and is_apropriate_lenght and rho1 - rho2 != 0:
            # Generate new extended peak
            new_peak_dict = {"ksi1": rho1,
                             "ksi2": rho2,
                             "beta": 0.5 * (theta1 + theta2),
                             "C_k": 0.5 * (acc_value1 + acc_value2)}
            extended_peaks.append(new_peak_dict)
            
    return extended_peaks


def vert_dist_is_valid(peak1, peak2, dist_t):
    ksi11, ksi12, beta1, c_1 = [peak1["ksi1"], peak1["ksi2"],
                                peak1["beta"], peak1["C_k"]]
    ksi21, ksi22, beta2, c_2 = [peak2["ksi1"], peak2["ksi2"],
                                peak2["beta"], peak2["C_k"]]
    ang_dif = abs(beta1 - beta2)

    vert_dist_cond1 = ((abs(ksi11 - ksi12) - c_1 * math.sin(ang_dif)) /
                       abs(ksi11 - ksi12))
    vert_dist_cond2 = ((abs(ksi21 - ksi22) - c_2 * math.sin(ang_dif)) /
                       abs(ksi21 - ksi22))
    
    if max([vert_dist_cond1, vert_dist_cond2]) < dist_t:
        return [True, ang_dif]

    return [False, ang_dif]


def find_valid_peaks_pair(peaks, dist_t):
    valid_peaks_pairs = []
    
    for peak1, peak2 in itertools.combinations(peaks, 2):
        condition, ang_dif = vert_dist_is_valid(peak1, peak2, dist_t)
        if condition:
            temp_dict = {"sides_a" : peak1,
                         "sides_b" : peak2,
                         "ang_dif" : ang_dif}
            valid_peaks_pairs.append(temp_dict)
            
    return valid_peaks_pairs


def gen_expected_perimeters(valid_peaks_pairs):
    for pair in valid_peaks_pairs:
        len_a = (abs(pair["sides_a"]["ksi1"] - pair["sides_a"]["ksi2"]) / 
                 math.sin(pair["ang_dif"]))
        len_b = (abs(pair["sides_b"]["ksi1"] - pair["sides_b"]["ksi2"]) / 
                 math.sin(pair["ang_dif"]))
        pair["exp_per"] = 2 * (len_a + len_b)


def validate_perimeter(valid_sides_pairs, actual_perimeter):
    best_pair = None
    min_delta = float("inf")
    for pair in valid_sides_pairs:
        val_par_condition = (abs(actual_perimeter - pair["exp_per"]) <
                             PERIMETER_T * pair["exp_per"])
        delta = (abs(abs(actual_perimeter - pair["exp_per"]) -
                     PERIMETER_T * pair["exp_per"]))
        if val_par_condition and delta < min_delta:
            best_pair = pair
            min_delta = delta
                  
    return best_pair


def get_sides_parameters(sides_pair):
    beta_a = sides_pair["sides_a"]["beta"]
    beta_b = sides_pair["sides_b"]["beta"]
    
    side1 = [sides_pair["sides_a"]["ksi1"], beta_a]
    side2 = [sides_pair["sides_a"]["ksi2"], beta_a]
    side3 = [sides_pair["sides_b"]["ksi1"], beta_b]
    side4 = [sides_pair["sides_b"]["ksi2"], beta_b]
    
    return [side1, side2, side3, side4]


def find_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    a_matrix = np.array([[math.cos(theta1), math.sin(theta1)],
                         [math.cos(theta2), math.sin(theta2)]])
    b_matrix = np.array([rho1, rho2])
    x_y = np.linalg.solve(a_matrix, b_matrix)
    return x_y



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
    actual_perimeter = np.shape(points)[0]
    
    rhos, thetas, ht_acc = hough_transform(points)
    # ht_acc_enh = enhance(ht_acc)
    rho_theta_acc = find_peaks_v2(ht_acc, rhos, thetas)
    
    extended_peaks = get_cooriented_pairs(rho_theta_acc, rhos, thetas)
    valid_peaks_pairs = find_valid_peaks_pair(extended_peaks, DIST_T)
    print("valid_peaks_len: ", len(valid_peaks_pairs))
    
    gen_expected_perimeters(valid_peaks_pairs)
    final_pairs = validate_perimeter(valid_peaks_pairs, actual_perimeter)
    #print("final_pairs type: ", type(final_pairs))
    
# =============================================================================
#     vertices = []
#     for pairs in valid_peaks_pairs: 
#         sides = get_sides_parameters(pairs)
#         x1_y1 = find_intersection(sides[0], sides[3])
#         x2_y2 = find_intersection(sides[0], sides[2])
#         x3_y3 = find_intersection(sides[1], sides[2])
#         x4_y4 = find_intersection(sides[1], sides[3])
#         vertices = [x1_y1, x2_y2, x3_y3, x4_y4]
#         
#         ring1 = LinearRing(points)
#         x1, y1 = ring1.xy
#         
#         ring2 = LinearRing(vertices)
#         x2, y2 = ring2.xy
#         
#         fig = pyplot.figure(1, figsize=(5, 5), dpi=90)
#         example = fig.add_subplot(111)
#         example.plot(x1, y1, marker='o')
#         example.set_title(FILE_PATH)
#         
#         ans = fig.add_subplot(111)
#         ans.plot(x2, y2)
#         ans.set_title(FILE_PATH)
# =============================================================================
    sides = get_sides_parameters(final_pairs)
    x1_y1 = find_intersection(sides[0], sides[3])
    x2_y2 = find_intersection(sides[0], sides[2])
    x3_y3 = find_intersection(sides[1], sides[2])
    x4_y4 = find_intersection(sides[1], sides[3])
    vertices = [x1_y1, x2_y2, x3_y3, x4_y4]
    
    ring1 = LinearRing(points)
    x1, y1 = ring1.xy
    
    ring2 = LinearRing(vertices)
    x2, y2 = ring2.xy
    
    fig = pyplot.figure(1, figsize=(5, 5), dpi=90)
    example = fig.add_subplot(111)
    example.plot(x1, y1, marker='o')
    example.set_title(FILE_PATH)
    
    ans = fig.add_subplot(111)
    ans.plot(x2, y2)
    ans.set_title(FILE_PATH)
    

run_algorithm(assign_figure(FILE_PATH))

# =============================================================================
# figure = assign_figure(FILE_PATH)
# 
# vertices = run_algorithm(figure)
# 
# ring1 = LinearRing(figure)
# x1, y1 = ring1.xy
# 
# ring2 = LinearRing(vertices)
# x2, y2 = ring2.xy
# 
# fig = pyplot.figure(1, figsize=(5, 5), dpi=90)
# example = fig.add_subplot(111)
# example.plot(x1, y1, marker='o')
# example.set_title(FILE_PATH)
# 
# ans = fig.add_subplot(111)
# ans.plot(x2, y2)
# ans.set_title(FILE_PATH)
# =============================================================================
