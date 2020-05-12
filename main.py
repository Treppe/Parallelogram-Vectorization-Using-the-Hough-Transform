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
MIN_ACCEPT_HEIGHT = np.array(3)     # Used in find_peaks()
LENGHT_T = 0.3                      # Used in get_paired_peaks() for coorientation validation step
DIST_T = 0.3                        # Used in vert_dist_is_valid() for distance validation step
PERIMETER_T = 0.1                   # Used in validate_perimeter() for perimeter validation step


# Choose figure to run
FILE_PATH = "Testing_Figures/1.txt"

def assign_figure(file_path):
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
    img = {"points": shape,
             "x_min": np.ceil(np.amin(shape[:, 0])),
             "x_max": np.ceil(np.amax(shape[:, 0])),
             "y_min": np.ceil(np.amin(shape[:, 1])),
             "y_max": np.ceil(np.amax(shape[:, 1])),
             "perimeter": Polygon(shape).length}
    img["height"] = img["y_max"] - img["y_min"]
    img["width"] = img["x_max"] - img["x_min"]
    return img


def create_rho_theta(hough_acc, img):
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
    x_min, x_max = [img["x_min"], img["x_max"]]
    y_min, y_max = [img["y_min"], img["y_max"]]    
    n_max = max(x_max - x_min, y_max - y_min)
    
    # Theta space
    hough_acc["d_theta"] = math.pi / (4*(n_max - 1))
    hough_acc["d_rho"] = math.pi / 8
    hough_acc["theta space"] = np.arange(-math.pi / 2, math.pi / 2, 
                                         hough_acc["d_theta"])
    
    # Rho space
    distance = np.sqrt((x_max + 1) ** 2 + (y_max + 1) ** 2)
    hough_acc["rho space"] = np.arange(-distance, distance, hough_acc["d_rho"])
    

def fill_hs_acc(hough_acc, points):
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
    theta_s = hough_acc["theta space"]
    rho_s = hough_acc["rho space"]
    
    for x, y in points:
        for theta_idx in range(len(theta_s)):
            rho_val = (x*math.cos(theta_s[theta_idx]) + 
                       y*math.sin(theta_s[theta_idx]))
            rho_idx = (np.nonzero(np.abs(rho_s - rho_val) ==
                       np.min(np.abs(rho_s - rho_val)))[0])
            hough_acc["accumulator"][rho_idx, theta_idx] += 1


def hough_transform(img):
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
    hough_acc = {}
    create_rho_theta(hough_acc, img)
    hough_acc["accumulator"] = np.zeros((len(hough_acc["rho space"]), 
                                         len(hough_acc["theta space"])))
    fill_hs_acc(hough_acc, img["points"])
    return hough_acc


def enhance(ht_acc, img_h, img_w):
    h = int(img_h) // 4
    w = int(img_w) // 4
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


def find_peaks(hough_acc, img):
    rho_theta_acc = []
    hough_acc_enh = enhance(hough_acc["accumulator"], 
                            img["height"], img["width"])
    
    while True:
        max_idx = np.where(hough_acc_enh == np.amax(hough_acc_enh))
        acc_value = hough_acc["accumulator"][max_idx[0], max_idx[1]][0]
        if acc_value >= MIN_ACCEPT_HEIGHT:
            rho = hough_acc["rho space"][max_idx[0]][0]
            theta = hough_acc["theta space"][max_idx[1]][0]
            rho_theta_acc.append({"rho": rho,
                                   "theta": theta,
                                   "acc value": acc_value})
            hough_acc_enh[max_idx[0], max_idx[1]] = 0
        else:
            break
            
    hough_acc["Hough peaks"] = rho_theta_acc


def get_paired_peaks(hough_acc):
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
    theta_t = 3*hough_acc["d_theta"]                                           # Theta threshold. Depends on theta space resolution
    for peak1, peak2 in itertools.combinations(hough_acc["Hough peaks"], 2):
        theta1, theta2 = [peak1["theta"], peak2["theta"]]
        rho1, rho2 = [peak1["rho"], peak2["rho"]]
        acc_value1, acc_value2 = [peak1["acc value"], peak2["acc value"]] 
        
        # Coorientation of sides conditions
        is_parallel = abs(theta1 - theta2) < theta_t
        is_apropriate_lenght = ((acc_value1 - acc_value2) <
                                LENGHT_T * (acc_value1 + acc_value2) * 0.5)
        
        if is_parallel and is_apropriate_lenght:
            # Generate new extended peak
            new_peak_dict = {"ksi1": rho1,
                             "ksi2": rho2,
                             "beta": 0.5 * (theta1 + theta2),
                             "acc value": 0.5 * (acc_value1 + 
                                                 acc_value2)}
            extended_peaks.append(new_peak_dict)
    assert len(extended_peaks) >= 2, \
            "At least 2 pairs must pass coorientation validation step. " + \
                "But only " + str(len(extended_peaks)) +  " were found." 
    return extended_peaks


def vert_dist_is_valid(peak1, peak2):
    ksi11, ksi12, beta1, c_1 = [peak1["ksi1"], peak1["ksi2"],
                                peak1["beta"], peak1["acc value"]]
    ksi21, ksi22, beta2, c_2 = [peak2["ksi1"], peak2["ksi2"],
                                peak2["beta"], peak2["acc value"]]
    ang_dif = abs(beta1 - beta2)
    
    if ksi11 - ksi12 != 0 and ksi21 - ksi22 != 0:
        vert_dist_cond1 = ((abs(ksi11 - ksi12) - c_1 * math.sin(ang_dif)) /
                           abs(ksi11 - ksi12))
        vert_dist_cond2 = ((abs(ksi21 - ksi22) - c_2 * math.sin(ang_dif)) /
                           abs(ksi21 - ksi22))
        
        if max([vert_dist_cond1, vert_dist_cond2]) < DIST_T:
            return [True, ang_dif]

    return [False, ang_dif]


def gen_parallelograms_sides(peaks_list):
    parallelograms_sides = []
    
    for peak1, peak2 in itertools.combinations(peaks_list, 2):
        condition, ang_dif = vert_dist_is_valid(peak1, peak2)
        if condition:
            temp_dict = {"sides_a" : peak1,
                         "sides_b" : peak2,
                         "ang_dif" : ang_dif}
            parallelograms_sides.append(temp_dict)
    
    assert len(parallelograms_sides) > 0, \
            "All sides combinations failed distance validation step."
    return parallelograms_sides


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
# =============================================================================
    image = gen_shape_dict(points)
    hough_acc = hough_transform(image)
    find_peaks(hough_acc, image)     
    paired_peaks = get_paired_peaks(hough_acc)
    parallelograms_sides = gen_parallelograms_sides(paired_peaks)
#     print("valid_peaks_len: ", len(valid_peaks_pairs))
#     
#     gen_expected_perimeters(valid_peaks_pairs)
#     final_pairs = validate_perimeter_v2(valid_peaks_pairs, actual_perimeter)
#     print("final_pairs type: ", type(final_pairs))
#     
#     sides = get_sides_parameters(final_pairs)
#     x1_y1 = find_intersection(sides[0], sides[3])
#     x2_y2 = find_intersection(sides[0], sides[2])
#     x3_y3 = find_intersection(sides[1], sides[2])
#     x4_y4 = find_intersection(sides[1], sides[3])
#     vertices = [x1_y1, x2_y2, x3_y3, x4_y4]
#     
#     ring1 = LinearRing(points)
#     x1, y1 = ring1.xy
#     
#     ring2 = LinearRing(vertices)
#     x2, y2 = ring2.xy
#     
#     fig = pyplot.figure(1, figsize=(5, 5), dpi=90)
#     example = fig.add_subplot(111)
#     example.plot(x1, y1, marker='o')
#     example.set_title(FILE_PATH)
#     
#     ans = fig.add_subplot(111)
#     ans.plot(x2, y2)
#     ans.set_title(FILE_PATH)
# =============================================================================
    return 0 

run_algorithm(assign_figure(FILE_PATH))


