'''
The algorithm which convert a noisy parallelogram 
into a set of four points - an ideal parallelogram.
'''

import numpy as np
import math
from matplotlib import pyplot
from shapely.geometry.polygon import LinearRing
from shapely.geometry import Polygon

# Parallelogram detecting thresholds
MIN_ACCEPT_HEIGHT = np.array(3)
LENGHT_T = 0.3
DIST_T = 0.55
PERIMETER_T = 0.0001

# Rho and theta discretive step (accumulator's resolution)
RHO_RES = 1
#RHO_RES = MIN_ACCEPT_HEIGHT / 15.0
#THETA_RES = 0.017453*10
THETA_RES = 0.0174533*RHO_RES # 0.0174533 rad = 1 grad
THETA_T = THETA_RES * 3
MIN_PEAKS_TO_FIND = 100

# Choose figure to run
FILE_PATH = "Testing_Figures/perfect.txt"


def assign_figure(file_path):
    file = open(file_path, "r")
    points = []
    for line in file:
        row = line.split()
        points.append([float(row[0]), float(row[1])])
    return np.array(points)


def find_max_points(points):
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
    
    x_max = np.ceil(np.amax(points[:, 0]))
    y_max = np.ceil(np.amax(points[:, 1]))
    
    return x_max, y_max

def create_rho_theta(x_max, y_max):
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
    theta = np.linspace(-math.pi / 2, 0.0, math.ceil(math.pi/ (2*THETA_RES) + 1))
    theta = np.concatenate((theta, -theta[len(theta)-2::-1]))

    D = np.sqrt((x_max + 10)**2 + (y_max + 10)**2)
    q = math.ceil(D/RHO_RES)
    nrho = 2*q + 1
    rho = np.linspace(-q*RHO_RES, q*RHO_RES, nrho)
    return theta, rho

def fill_hs_acc(ht_acc, points, rho, theta):
    """
    Compute voices for every (rho, theta) pair in Hough Accumulator and update it values

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
    for x, y in points:
        for thIdx in range(len(theta)):
            rhoVal = x*math.cos(theta[thIdx]) + \
                    y*math.sin(theta[thIdx])
            rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
            ht_acc[rhoIdx, thIdx] += 1
    #ht_acc[ht_acc < MIN_ACCEPT_HEIGHT] = 0
    return ht_acc           

def hough_transform(points):
    """
    Builds a Hough H(theta, rho) space for given arrai of (x,y) coordinates
    
    Parameters
    ----------
    points : np.array
        2D array of points to transform
    theta_res : float
        Theta resolution. Determines discretisation step for theta. The default is 1.0.
    rho_res : TYPE, float
        Rho resolution. Determines discretisation step for rho. The default is 1.0
    enhanced : bool
        Determine if normal or encanced accumulator is required. The default is True
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
    x_max, y_max = find_max_points(points)
    theta, rho = create_rho_theta(x_max, y_max)
    ht_acc = np.zeros((len(rho), len(theta)))
    fill_hs_acc(ht_acc, points, rho, theta)
    return theta, rho, ht_acc

def find_peaks(ht_acc, rhos, thetas):
    rho_theta_acc = set([])
    mask_height = 1
    mask_width = 1
    while len(rho_theta_acc) < MIN_PEAKS_TO_FIND:
        acc_max = np.amax(ht_acc)
        acc_h, acc_w = np.shape(ht_acc) 
        if acc_max != 0:
            peak_idx_list = np.argwhere(ht_acc == acc_max) # Get an index of the highest peak
            for peak_idx in peak_idx_list:
                acc_value = ht_acc[peak_idx[0], peak_idx[1]]
                rho = rhos[peak_idx[0]]
                theta = thetas[peak_idx[1]]
                mask_origin = np.array([(peak_idx[0] - mask_height // 2), 
                               (peak_idx[1] - mask_width // 2)])
                mask_origin[mask_origin < 0] = 0
                
                ht_acc[mask_origin[0] : mask_origin[0] + mask_height, 
                mask_origin[1] : mask_origin[1] + mask_width] = 0
                rho_theta_acc.add((rho, theta, acc_value))
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
        Angular threshold that determines if peaks correspond to a parallel line
    len_T : float, optional
        Normolized threshold that verifies if line segment corresponding
        to pair of peaks have approximately the same length. 
        The default is 0.5.

    Returns
    -------
    satisfying_pairs : list
        Pairs of peaks occuring at the same orientation theta, and with similar heights.

    """
    
    extended_peaks = []
    for current_peak in peaks[:-1]:
        cur_idx = peaks.index(current_peak)
        for compare_peak in peaks[cur_idx + 1:]:
            rho1, theta1, acc_value1 = current_peak
            rho2, theta2, acc_value2 = compare_peak
            is_parallel = abs(theta1 - theta2) < THETA_T
            is_apropriate_lenght = abs(acc_value1- acc_value2) < LENGHT_T * (acc_value1 + acc_value2) * 0.5
            if is_parallel and is_apropriate_lenght:
                # Create new extended peak
                temp_dict = {"ksi1":rho1,
                             "ksi2":rho2,
                             "beta":0.5 * (theta1 + theta2),
                             "C_k":0.5 * (acc_value1 + acc_value2)}
                extended_peaks.append(temp_dict)
    return extended_peaks

def vert_dist_is_valid(peak1, peak2):
    ksi11, ksi12, beta1, C1 = [peak1["ksi1"], peak1["ksi2"], peak1["beta"], peak1["C_k"]]
    ksi21, ksi22, beta2, C2 = [peak2["ksi1"], peak2["ksi2"], peak2["beta"], peak2["C_k"]]
    ang_dif = abs(beta1 - beta2)
    
    if abs(ksi11 - ksi12) < MIN_ACCEPT_HEIGHT or abs(ksi21 - ksi22) < MIN_ACCEPT_HEIGHT:
        return [False, ang_dif]
    
    vert_dist_cond1 = (abs(ksi11 - ksi12) - C1 * math.sin(ang_dif)) / abs(ksi11 - ksi12)
    vert_dist_cond2 = (abs(ksi21 - ksi22) - C2 * math.sin(ang_dif)) / abs(ksi21 - ksi22)
    if max([vert_dist_cond1 , vert_dist_cond2]) < DIST_T:
        return [True, ang_dif]

    return [False, ang_dif]

def find_valid_peaks_pair(peaks):
    answer = []
    for current_peak in peaks[:-1]:
        cur_idx = peaks.index(current_peak)
        for other_peak in peaks[cur_idx + 1:]:
            condition, ang_dif = vert_dist_is_valid(current_peak, other_peak)
            if condition:
                output = [current_peak, other_peak, ang_dif]
                answer.append(output)
    return answer

def find_intersection(line1, line2):  
    rho1, theta1 = line1
    rho2, theta2 = line2
    theta1 = theta1
    theta2 = theta2
    a_matrix = np.array([[math.cos(theta1), math.sin(theta1)], [math.cos(theta2), math.sin(theta2)]])
    b_matrix = np.array([rho1, rho2])
    x_y = np.linalg.solve(a_matrix, b_matrix)
    return x_y

def gen_expected_perimeters(valid_peaks_pairs, len_factor):
    for pair in valid_peaks_pairs:
        len_a = abs(pair[0]["ksi1"] - pair[0]["ksi2"]) / math.sin(pair[2])
        len_b = abs(pair[1]["ksi1"] - pair[1]["ksi2"]) / math.sin(pair[2])
        pair.append({"exp_per" : 2 * (len_a + len_b)})

def gen_actual_perimeter(edge_points):
    return Polygon(edge_points).length

def validate_perimeter(valid_sides_pairs, actual_perimeter):
    best_pair = None
    min_delta = float("inf")
    for pair in valid_sides_pairs:
        val_par_condition = abs(actual_perimeter - pair[3]["exp_per"]) < PERIMETER_T * pair[3]["exp_per"]
        delta = abs(abs(actual_perimeter - pair[3]["exp_per"]) - PERIMETER_T * pair[3]["exp_per"])
        if val_par_condition and delta < min_delta:
            best_pair = pair
            min_delta = delta
    return best_pair
       
def get_sides_parameters(sides_pair):
    side1 = [sides_pair[0]["ksi1"], sides_pair[0]["beta"]]
    side2 = [sides_pair[0]["ksi2"], sides_pair[0]["beta"]]
    side3 = [sides_pair[1]["ksi1"], sides_pair[1]["beta"]]
    side4 = [sides_pair[1]["ksi2"], sides_pair[1]["beta"]]
    return [side1, side2, side3, side4]
     
def run_algorithm(figure):
    '''
    

    Parameters
    ----------
    figure : TYPE, string
        DESCRIPTION. Name of figure to detect
    enhanced : bool
        Determine if normal or encanced accumulator is required. The default is True
    
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
        DESCRIPTION. Ordered list of top rho and theta pairs (has the most value in Hough accumulator)
    x_y_pairs : np.array
        DESCRIPTION. x and y coordinates corresponding to sinusoids in Hough Space
    
    '''
    global DIST_T, PERIMETER_T
    actual_perimeter = gen_actual_perimeter(figure)
    edge_perimeter = np.shape(figure)[0]
    len_factor = actual_perimeter / edge_perimeter
    DIST_T = DIST_T * len_factor
    thetas, rhos, ht_acc = hough_transform(figure)
    rho_theta_acc = find_peaks(ht_acc, rhos, thetas)
    extended_peaks = get_cooriented_pairs(rho_theta_acc, rhos, thetas)
    valid_peaks_pairs = find_valid_peaks_pair(extended_peaks)
    print (len(valid_peaks_pairs))
    gen_expected_perimeters(valid_peaks_pairs, len_factor)
    final_pairs = validate_perimeter(valid_peaks_pairs, actual_perimeter)
    print (type(final_pairs))
    sides = get_sides_parameters(final_pairs)
    x1_y1 = find_intersection(sides[0], sides[3])
    x2_y2 = find_intersection(sides[0], sides[2])
    x3_y3 = find_intersection(sides[1], sides[2])
    x4_y4 = find_intersection(sides[1], sides[3])
    vertices = [x1_y1, x2_y2, x3_y3, x4_y4]

    return vertices


#================================================TEST CASES=======================================================

figure = assign_figure(FILE_PATH)

vertices = run_algorithm(figure)

ring1 = LinearRing(figure)
x1, y1 = ring1.xy

#ring2 = LinearRing(vertices)
#x2, y2= ring2.xy

fig = pyplot.figure(1, figsize=(5,5), dpi=90)
example = fig.add_subplot(111)
example.plot(x1, y1, marker = 'o')
example.set_title(FILE_PATH)

#ans = fig.add_subplot(111)
#ans.plot(x2, y2)
#ans.set_title(FILE_PATH)


#print (figure)
    