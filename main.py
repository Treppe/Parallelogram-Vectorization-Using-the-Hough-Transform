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
# =============================================================================
# <<<<<<< Updated upstream
# MIN_ACCEPT_HEIGHT = np.array(4)
# =======
MIN_ACCEPT_HEIGHT = np.array(3)
# >>>>>>> Stashed changes
# =============================================================================
LENGHT_T = 0.5
DIST_T = 0.5
PERIMETER_T = 0.1

# =============================================================================
# # Rho and theta discretive step (accumulator's resolution)
# RHO_RES = 1
# #RHO_RES = 0.1 * MIN_ACCEPT_HEIGHT
# # THETA_RES = 0.017453*10
# THETA_RES = 0.01745331 # 0.0174533 rad = 1 grad
# THETA_T = THETA_RES * 3
# MIN_PEAKS_TO_FIND = 100
# 
# # MASK CONSTATS
# MASK_HEIGHT = 1
# MASK_WIDTH = 1
# 
# # Image Enhencement Constants
# ENH_H = 5
# ENH_W = 5
# =============================================================================

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
    img_height = y_max - y_min
    img_width = x_max - x_min
    
    n_max = max(x_max - x_min, y_max - y_min)
    
    d_theta = math.pi / (4*(n_max - 1))
    THETA_T = 3 * d_theta
    d_rho = math.pi / 8
     
    theta = np.arange(-math.pi / 2, math.pi / 2 + d_theta, d_theta)
     
    distance = np.sqrt((x_max + 1)**2 + (y_max + 1)**2)
    rho = np.arange(-distance, distance, d_rho)
# =============================================================================
# =============================================================================
#     theta = np.linspace(-math.pi / 2, 0.0,
#                          math.ceil(math.pi / (2*THETA_RES) + 1))
#     theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
#   
#     distance = np.sqrt((x_max + 1)**2 + (y_max + 1)**2)
#     dummy = math.ceil(distance / RHO_RES)
#     nrho = 2*dummy + 1
#     rho = np.linspace(-dummy*RHO_RES, dummy*RHO_RES, nrho)
# =============================================================================
# =============================================================================
    
    return rho, theta, img_height, img_width, d_rho, d_theta


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
    # ht_acc[ht_acc < MIN_ACCEPT_HEIGHT] = 0
    
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
    
    
    rho, theta, img_height, img_width, d_rho, d_theta = create_rho_theta(points)
    ht_acc = np.zeros((len(rho), len(theta)))
    fill_hs_acc(ht_acc, points, rho, theta)
    return rho, theta, ht_acc, img_height, img_width, d_rho, d_theta

def enhance(ht_acc, img_h, img_w, d_rho, d_theta):
    h = int(img_h) // 4
    w = int(img_w) // 4
    #h = d_rho // (math.sqrt((img_h / 2)**2 + (img_w / 2)**2))
    #w = d_theta // (5 * math.pi / img_w)
    ht_acc_enh = np.copy(ht_acc)
    for row in range(ht_acc.shape[0]):
        if np.sum(ht_acc[row, :]) != 0:
            for col in range(ht_acc.shape[1]):
                mask_origin = np.array([row - h // 2, col - w//2])
                mask_origin[mask_origin < 0] = 0
                integral = np.sum(ht_acc_enh[mask_origin[0] : mask_origin[0] + 
                                             h + 1, 
                                             mask_origin[1] : mask_origin[1] +
                                             w + 1])
                if integral != 0:
                    ht_acc_enh[row, col] = (h * w * ht_acc[row, col]**2 / 
                                            integral) 
    return ht_acc_enh



def find_peaks(ht_acc, rhos, thetas, img_h, img_w, d_rho, d_theta):
    rho_theta_acc = []
    ht_acc_enh = enhance(ht_acc, img_h, img_w, d_rho, d_theta)
    while True:
        max_idx = np.where(ht_acc_enh == np.amax(ht_acc_enh))
        h_peak = ht_acc[max_idx[0], max_idx[1]][0]  # Always returns only 1 point
        if h_peak >= MIN_ACCEPT_HEIGHT:
            rho = rhos[max_idx[0]][0]
            theta = thetas[max_idx[1]][0]
            print(h_peak, rho, theta)
            rho_theta_acc.append((rho, theta, h_peak))
            ht_acc_enh[max_idx[0], max_idx[1]] = 0
        else:
            break
            
    return rho_theta_acc


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
        
        if is_parallel and is_apropriate_lenght:
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
    
    if ksi11 - ksi12 != 0 and ksi21 - ksi22 != 0:
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
    min_delta = float('inf')
    for pair in valid_sides_pairs:
        val_par_condition = (abs(actual_perimeter - pair["exp_per"]) <
                              PERIMETER_T * pair["exp_per"])
        if val_par_condition:
            delta = (abs(abs(actual_perimeter - pair["exp_per"]) -
                         PERIMETER_T * pair["exp_per"]))
            if delta < min_delta:
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

def validate_perimeter_v2(valid_sides_pairs, actual_perimeter):
    min_dif = float("inf")
    best_pair = None
    for pair in valid_sides_pairs:
        sides = get_sides_parameters(pair)
        x1_y1 = find_intersection(sides[0], sides[3])
        x2_y2 = find_intersection(sides[0], sides[2])
        x3_y3 = find_intersection(sides[1], sides[2])
        x4_y4 = find_intersection(sides[1], sides[3])
        vertices = [x1_y1, x2_y2, x3_y3, x4_y4]
        exp_per = Polygon(vertices).length
        per_dif = abs(actual_perimeter - exp_per)
        if per_dif < min_dif:
            min_dif = per_dif
            best_pair = pair
        
    return best_pair
    
    
def standart_deviastion(points_arr, figure_sides):
    for point in points_arr:
        for side in figure_sides:
             pt_lin_distance = abs(point[0]*math.cos(side[1]) +
                                   point[1]*math.sin(side[1]) -
                                   - side[0])	                                  

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
    actual_perimeter = Polygon(points).length
    #actual_perimeter = np.shape(points)[0]    
    rhos, thetas, ht_acc, img_height, img_width, d_rho, d_theta = hough_transform(points)
    # Enhance test
    #ht_acc_enh = enhance(ht_acc, img_height, img_width, d_rho, d_theta)
    
    #rhos, thetas, ht_acc = hough_transform(points)
    # ht_acc_enh = enhance(ht_acc)
    rho_theta_acc = find_peaks(ht_acc, rhos, thetas, img_height, img_width, d_rho, d_theta)
    
    extended_peaks = get_cooriented_pairs(rho_theta_acc, rhos, thetas)
    valid_peaks_pairs = find_valid_peaks_pair(extended_peaks, DIST_T)
    print("valid_peaks_len: ", len(valid_peaks_pairs))
    
    gen_expected_perimeters(valid_peaks_pairs)
    final_pairs = validate_perimeter_v2(valid_peaks_pairs, actual_perimeter)
    print("final_pairs type: ", type(final_pairs))
    
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
