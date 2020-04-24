'''
The algorithm which convert a noisy parallelogram 
into a set of four points - an ideal parallelogram.
'''

import numpy as np
import math
from matplotlib import pyplot
from shapely.geometry.polygon import LinearRing


# Assign rho and theta discretive step (accumulator's resolution)

# Assign parallelogram detecting thresholds
MIN_ACCEPT_HEIGHT = np.array(3)
LENGHT_T = 0.3
DIST_T = 0.6
APROX_WIDTH = 0.1
PERIMETER_T = 0.3


RHO_RES = MIN_ACCEPT_HEIGHT / 10.0
THETA_RES = 0.0174533*RHO_RES

# Accumulator enhansing constants
ENH_AREA_HEIGHT = 134
ENH_AREA_WIDTH = 175
ENH_MIN_ACCEPT_HEIGHT = np.array(ENH_AREA_HEIGHT * ENH_AREA_WIDTH)
# Choose figure to run
FIGURE = "example 1"
print ("START_MIN_ACCEPT_HEIGHT: ", MIN_ACCEPT_HEIGHT)
print ("LENGHT_T: ", LENGHT_T)
print ("DIST_T: ", DIST_T)
print ("FIGURE: ", FIGURE)


test_counter = 0

def assign_figure(figure_name):
    '''
    
    Parameters
    ----------
    figure_name : TYPE, string
        DESCRIPTION. Name from possible figures list.

    Returns
    -------
    np.array
        2D array of points representing required figure

    POSSIBLE FIGURES LIST:
    example 1
    example 2
    example 3
    line
    square
    '''
    points = None
    if figure_name == "example 1": 
        points = np.array([[8.04411, 78.9279], 
                        [8.89507, 79.1510],
                        [9.75144, 79.5972],
                        [11.0208, 79.4857],
                        [11.4095, 78.0355],
                        [11.8096, 77.1432],
                        [12.2081, 76.2508],
                        [12.4008, 75.5815],
                        [11.5546, 75.3584],
                        [10.2843, 74.9122],
                        [9.43571, 74.4660],
                        [8.1698, 74.01980],
                        [7.32486, 73.5736],
                        [6.48152, 73.1274],
                        [5.22202, 72.6812],
                        [4.38229, 72.2350],
                        [2.71349, 72.3466],
                        [2.3010, 73.3505],
                        [1.88804, 74.6891],
                        [1.47164, 75.6930],
                        [1.68387, 76.2508],
                        [2.94818, 76.4739],
                        [3.79413, 76.9201],
                        [4.64168, 77.3663],
                        [5.91321, 77.8125],
                        [6.76436, 78.2586], 
                        [8.0422, 78.8164]])
   
    elif figure_name == "example 2": # 0.5 0.5 0.3 0.3
        points = np.array([[17.5544, 72.9043],
                         [18.2031, 73.462],
                         [18.6617, 74.466],
                         [19.0994, 74.9122],
                         [19.6522, 75.5815],
                         [20.8082, 75.5815],
                         [21.2083, 75.1353],
                         [21.6076, 74.6891],
                         [22.4253, 74.2429],
                         [22.8226, 73.7967],
                         [23.2191, 73.3505],
                         [23.6148, 72.9043],
                         [24.0098, 72.4581],
                         [24.833, 72.235],
                         [24.5888, 71.5657],
                         [24.1255, 70.6733],
                         [23.6867, 70.2271],
                         [23.2487, 69.7809],
                         [22.8115, 69.3347],
                         [22.3535, 68.4424],
                         [21.9183, 67.9962],
                         [21.4839, 67.55],
                         [21.0503, 67.1038],
                         [20.5974, 66.2114],
                         [20.1658, 65.7652],
                         [19.735, 65.319],
                         [19.305, 64.8728],
                         [19.0624, 63.9804],
                         [18.0463, 64.2035],
                         [17.2429, 64.6497],
                         [16.8488, 65.0959],
                         [16.4539, 65.5421],
                         [16.0582, 65.9883],
                         [15.2495, 66.4345],
                         [14.8518, 66.8807],
                         [14.4533, 67.3269],
                         [14.0539, 67.7731],
                         [13.6505, 68.1077],
                         [14.0744, 68.4424],
                         [14.7131, 69.0001],
                         [15.1605, 70.004],
                         [15.5909, 70.4502],
                         [16.0221, 70.8964],
                         [16.4541, 71.3426],
                         [16.9031, 72.235],
                         [17.4457, 72.7928]])
    
    elif figure_name == "example 3":
        points = np.array([[9.45637, 65.319],
                         [10.2887, 65.7652],
                         [11.1226, 66.2114],
                         [11.9581, 66.6576],
                         [12.7953, 67.1038],
                         [13.4209, 67.3269],
                         [13.6009, 66.4345],
                         [13.9858, 65.5421],
                         [14.3691, 64.6497],
                         [14.7508, 63.7573],
                         [15.1308, 62.8649],
                         [15.5245, 62.4187],
                         [15.8979, 61.4148],
                         [15.6748, 60.857],
                         [14.8532, 60.6339],
                         [14.0255, 60.1877],
                         [13.1994, 59.7415],
                         [12.7806, 59.2953],
                         [11.9573, 58.8492],
                         [11.1356, 58.403],
                         [10.3156, 57.9568],
                         [9.49707, 57.5106],
                         [8.68019, 57.0644],
                         [8.26824, 56.6182],
                         [7.25631, 56.3951],
                         [7.06878, 57.2875],
                         [6.67808, 58.1799],
                         [6.28577, 59.0723],
                         [5.88605, 59.5184],
                         [5.49094, 60.4108],
                         [5.09423, 61.3032],
                         [4.59495, 62.3072],
                         [5.31363, 62.6418],
                         [5.72799, 63.088],
                         [6.55269, 63.5342],
                         [7.37899, 63.9804],
                         [8.20689, 64.4266],
                         [9.0364, 64.8728]])
   
    elif figure_name == "line":
        points = np.array([[1,1],
                 [2,2],
                 [3,3],
                 [4,4]])
        points = points * 10
    
    elif figure_name == "square": 
        points = np.array([[10,10],
                           [10,15],
                           [10,20],
                           [10,25],
                           [10,30],
                           [15,30],
                           [20,30],
                           [25,30],
                           [30,30],
                           [30,25],
                           [30,20],
                           [30,15],
                           [30,10],
                           [25,10],
                           [20,10]])
        
    elif figure_name == "acute":
            points = np.array([[0.000000, 0.000000],
                                [6.263796, 5.790209],
                                [16.504183, 12.045964],
                                [24.081099, 20.020319],
                                [41.495261, 32.706964],
                                [50.777048, 41.809076],
                                [57.490079, 45.314233],
                                [64.352872, 50.723552],
                                [75.10971, 58.023582],
                                [86.551405, 69.207839],
                                [94.782589, 76.523081],
                                [101.366271, 74.227111],
                                [109.831728, 83.838324],
                                [131.510132, 105.9979],
                                [147.695705, 114.980673],
                                [176.869812, 140.862549],
                                [182.696924, 147.721467],
                                [182.696924, 153.721467],
                                [184.557642, 158.362343],
                                [189.076471, 162.309518],
                                [162.309518, 162.309518],
                                [197.877231, 182.877386],
                                [200.325566, 191.537966],
                                [207.2482, 200.086483],
                                [209.011756, 203.67673],
                                [214.23023, 217.739714],
                                [223.023281, 231.106922],
                                [223.699273, 236.061015],
                                [225.166413, 240.84092],
                                [230.963049, 249.049022],
                                [249.049022, 249.049022],
                                [243.297775, 273.718474],
                                [228.064731, 268.527873],
                                [211.620993, 254.439087],
                                [200.325566, 249.692029],
                                [192.365551, 244.944971],
                                [172.733471, 231.106922],
                                [156.389932, 214.584216],
                                [151.82272, 203.67673],
                                [143.812574, 201.881606],
                                [138.159218, 192.976421],
                                [110.819352, 171.123921],
                                [104.2386, 162.309518],
                                [92.809975, 152.125433],
                                [83.719461, 144.292008],
                                [82.11803, 139.399763],
                                [66.427963, 132.855926],
                                [59.58997, 121.543927],
                                [59.58997, 116.543927],
                                [58.342477, 114.980673],
                                [57.290815, 112.171046],
                                [54.462976, 111.169382],
                                [54.462976, 104.169382],
                                [43.699884, 93.721615],
                                [41.013129, 82.026259],
                                [37.113555, 71.574606],
                                [24.906223, 54.236195],
                                [20.05031, 34.834646],
                                [6.263796, 18.993811]])
    return points

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
    # Create theta dimension
    #n_max = max(x_max, y_max)
    #d_theta = math.pi / (2*(n_max - 1))
    #theta = np.arange(-math.pi/2, math.pi/2, d_theta)
    # Create rho dimension
    # Max rho length is the distance to the farrest possible (x,y) point 
    # in given points array
    #distance = np.sqrt((x_max - 1)**2 + (y_max - 1)**2)
    #d_rho = math.pi/4 # Num of rho steps between min and max rho
    #rho = np.arange(-distance, distance, d_rho)
    theta = np.linspace(-math.pi / 2, 0.0, math.ceil(math.pi/ (2*THETA_RES) + 1))
    theta = np.concatenate((theta, -theta[len(theta)-2::-1]))

    D = np.sqrt((x_max - 1)**2 + (y_max - 1)**2)
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
    ht_acc[ht_acc < MIN_ACCEPT_HEIGHT] = 0
    return ht_acc           


def enhance_hs_acc(ht_acc, rho, theta):
    """

    Parameters
    ----------
    ht_acc : np.array
        Hough transform accumulator matrix C (rho by theta)
    rho : np.array
        Ordered array of rhos represented by rows in C
    theta : np.array
        Ordered array of thetas represented by columns in C

    Returns
    -------
    ht_acc_enh : np.array
        Enhansed version of ht_acc used to extract highest peaks more easily
    """
    h = 3
    w = math.ceil(5 * math.pi / 180)
    ht_acc_enh = np.array(ht_acc)
    indxes = np.argwhere(ht_acc_enh >= MIN_ACCEPT_HEIGHT)
    for row_col in indxes:
        mask_origin = ((row_col[0] - h) % h, (row_col[1] - w) % w)
        integer = np.sum(ht_acc_enh[mask_origin[0] : mask_origin[0] + h, mask_origin[1] : mask_origin[1] + w])
        if integer != 0:
            ht_acc_enh[row_col[0], row_col[1]] = h * w *  ht_acc_enh[row_col[0], row_col[1]] ** 2 / integer
    return ht_acc_enh

def find_peaks_v2(ht_acc, rhos, thetas):
    rho_theta_acc = []
    mask_height = 2
    mask_width = 10 # 5 grads equivalent
    while len(rho_theta_acc) < 50:
        peak_idx_list = np.argwhere(ht_acc == np.amax(ht_acc)) # Get an index of the highest peak
        for peak_idx in peak_idx_list:
            acc_value = ht_acc[peak_idx[0], peak_idx[1]]
            if acc_value != 0:
                rho = rhos[peak_idx[0]]
                theta = thetas[peak_idx[1]]
                mask_origin = [peak_idx[0] - mask_height // 2, peak_idx[1] - mask_width // 2]
                ht_acc[mask_origin[0] : mask_origin[0] + mask_height, 
                           mask_origin[1] : mask_origin[1] + mask_width] = 0
                rho_theta_acc.append([rho, theta, acc_value])
    return rho_theta_acc

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
    #ht_acc_enh = enhance_hs_acc(ht_acc, rho, theta)
    return theta, rho, ht_acc

def top_n_rho_theta_pairs(ht_acc_matrix, n, rhos, thetas):
  '''
  @param hough transform accumulator matrix H (rho by theta)
  @param n pairs of rho and thetas desired
  @param ordered array of rhos represented by rows in H
  @param ordered array of thetas represented by columns in H
  @return top n rho theta pairs in H by accumulator value
  @return x,y indexes in H of top n rho theta pairs
  '''
  flat = list(set(np.hstack(ht_acc_matrix)))
  flat_sorted = sorted(flat, key = lambda n: -n)
  coords_sorted = [(np.argwhere(ht_acc_matrix == acc_value)) for acc_value in flat_sorted[0:n]]
  rho_theta = []
  x_y = []
  for coords_for_val_idx in range(0, len(coords_sorted), 1):
    coords_for_val = coords_sorted[coords_for_val_idx]
    for i in range(0, len(coords_for_val), 1):
      k,m = coords_for_val[i] # n by m matrix
      rho = rhos[k]
      theta = thetas[m]
      rho_theta.append([rho, theta])
      x_y.append([k, n]) # just to unnest and reorder coords_sorted
  return rho_theta[0:n]

def find_peaks(ht_acc_enh, rhos, thetas):
    '''
    

    Parameters
    ----------
    ht_acc : np.array
        Hough transform accumulator matrix C (rho by theta)

    rhos : list
        Ordered array of rhos represented by rows in C
    thetas : list
        Ordered array of thetas represented by columns in C

    Returns
    -------
    rho_theta : list
        List of rho and theta parameters for lines which got at least MIN_ACCEPT_HEIGHT voices
        '''
    rho_theta = []
    flat = list(set(np.hstack(ht_acc_enh)))
    flat = np.delete(flat, np.argwhere(flat < MIN_ACCEPT_HEIGHT))
    flat_sorted = sorted(flat)
    coords_sorted = [(np.argwhere(ht_acc_enh == acc_value)) for acc_value in flat_sorted]
    rho_theta = []
    for coords_for_val_idx in range(0, len(coords_sorted), 1):
      coords_for_val = coords_sorted[coords_for_val_idx]
      for idx in range(0, len(coords_for_val), 1):
        k,m = coords_for_val[idx] # k by m matrix
        rho = rhos[k]
        theta = thetas[m]
        rho_theta.append([rho, theta])
    return rho_theta

def peaks_to_dict(peak_pairs):
    peaks_dict_list = []
    temp_dict = {"rhos": None, "thetas": None, "acc_values": None}
    for pair in peak_pairs:
        temp_dict = {"rhos": [pair[0][0], pair[2][0]], 
                     "thetas": [pair[0][1], pair[2][1]], 
                     "acc_values": [pair[1], pair[3]]}
        peaks_dict_list.append(temp_dict)
    return peaks_dict_list

def get_cooriented_pairs(peaks, rhos, thetas,):
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
    theta_t = THETA_RES * 3
    extended_peaks = []
    for current_peak in peaks[:-1]:
        cur_idx = peaks.index(current_peak)
        for compare_peak in peaks[cur_idx + 1:]:
            rho1, theta1, acc_value1 = current_peak
            rho2, theta2, acc_value2 = compare_peak
            is_parallel = abs(theta1 - theta2) < theta_t
            is_apropriate_lenght = abs(acc_value1- acc_value2) < LENGHT_T * (acc_value1 + acc_value2) * 0.5
            if is_parallel and is_apropriate_lenght:
                # Create new extended peak
                temp_dict = {"ksi1":rho1,
                             "ksi2":rho2,
                             "beta":0.5 * (theta1 + theta2),
                             "C_k":0.5 * (acc_value1 + acc_value2)}
                extended_peaks.append(temp_dict)
    return extended_peaks

def gen_extended_peaks(peak_pairs):
    """
    

    Parameters
    ----------
    pairs : list
        Dict of chosen pairs of peaks from Hough Accumulator and their parameters.

    Returns
    -------
    extended_peaks : list
        List of 2 lists of parameters required for each peak for further parallelogram sides detecting. 

    """
    extended_peaks = []
    for pair in peak_pairs:
        ksi1, ksi2 = pair["rhos"]
        acc_value1, acc_value2 = pair["acc_values"]
        beta = 0.5 * (pair["thetas"][0] + pair["thetas"][1])
        acc_value_extended = 0.5 * (acc_value1 + acc_value2)
        temp_dict = {"ksi1":ksi1,
                     "ksi2":ksi2,
                     "beta":beta,
                     "C_k":acc_value_extended}
        extended_peaks.append(temp_dict)
    return extended_peaks

def vert_dist_is_valid(peak1, peak2):
    ksi11, ksi12, beta1, C1 = [peak1["ksi1"], peak1["ksi2"], peak1["beta"], peak1["C_k"]]
    ksi21, ksi22, beta2, C2 = [peak2["ksi1"], peak2["ksi2"], peak2["beta"], peak2["C_k"]]
    ang_dif = abs(beta1 - beta2)
    if abs(ksi11 - ksi12) < MIN_ACCEPT_HEIGHT or abs(ksi21 - ksi22) < MIN_ACCEPT_HEIGHT:
        #print((ksi11 - ksi12) - MIN_ACCEPT_HEIGHT)
        #print ((ksi21 - ksi22) - MIN_ACCEPT_HEIGHT)
        #print ()
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

def gen_expected_perimeters(valid_peaks_pairs):
    for pair in valid_peaks_pairs:
        len_a = abs(pair[0]["ksi1"] - pair[0]["ksi2"]) / math.sin(pair[2])
        len_b = abs(pair[1]["ksi1"] - pair[1]["ksi2"]) / math.sin(pair[2])
        pair.append({"exp_per" : 2 * (len_a + len_b)})    

def get_sides_parameters(sides_pair):
    side1 = [sides_pair[0]["ksi1"], sides_pair[0]["beta"]]
    side2 = [sides_pair[0]["ksi2"], sides_pair[0]["beta"]]
    side3 = [sides_pair[1]["ksi1"], sides_pair[1]["beta"]]
    side4 = [sides_pair[1]["ksi2"], sides_pair[1]["beta"]]
    return [side1, side2, side3, side4]

def gen_actual_perimeters(valid_peaks_pairs, points):
    for pair in valid_peaks_pairs:
        actual_perimeter = 0
        # Get sides line equation parameters: rho, theta
        sides_list = get_sides_parameters(pair)
        # Count points near each side
        for side in sides_list:
            counted_points = []
            # Get distance between point and side line
            for point in points:
                pt_lin_distance = abs(point[0]*math.cos(side[1]) +
                                     point[1]*math.sin(side[1]) -
                                     side[0])
                if pt_lin_distance <= APROX_WIDTH and list(point) not in counted_points:
                    actual_perimeter += 1
                    counted_points.append(list(point))
        pair[3].update({"act_per" : actual_perimeter})

def validate_perimeter(valid_sides_pairs):
    best_pair = None
    min_delta = float("inf")
    for pair in valid_sides_pairs:
        val_par_condition = abs(pair[3]["act_per"] - pair[3]["exp_per"]) < PERIMETER_T * pair[3]["exp_per"]
        delta = abs(abs(pair[3]["act_per"] - pair[3]["exp_per"]) - PERIMETER_T * pair[3]["exp_per"])
        if val_par_condition and delta < min_delta:
            best_pair = pair
            min_delta = delta
    return best_pair
            
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
    figure = assign_figure(figure)
    
    #while valid_peaks == None and cur_min_accept_height  != np.array(3):
    thetas, rhos, ht_acc = hough_transform(figure)
    rho_theta_acc= find_peaks_v2(ht_acc, rhos, thetas)
    #rho_theta_pairs = top_n_rho_theta_pairs(ht_acc, 10, rhos, thetas)
    extended_peaks = get_cooriented_pairs(rho_theta_acc, rhos, thetas)
    valid_peaks_pairs = find_valid_peaks_pair(extended_peaks)
    gen_expected_perimeters(valid_peaks_pairs)
    gen_actual_perimeters(valid_peaks_pairs, figure)
    
    final_pairs = validate_perimeter(valid_peaks_pairs)
    sides = get_sides_parameters(final_pairs)
    x1_y1 = find_intersection(sides[0], sides[3])
    x2_y2 = find_intersection(sides[0], sides[2])
    x3_y3 = find_intersection(sides[1], sides[2])
    x4_y4 = find_intersection(sides[1], sides[3])
    vertices = [x1_y1, x2_y2, x3_y3, x4_y4]

    return thetas, rhos, ht_acc, rho_theta_acc, extended_peaks, valid_peaks_pairs, vertices


#================================================TEST CASES=======================================================
thetas, rhos, ht_acc, rho_theta_pairs, extended_peaks, valid_peaks, vertices = run_algorithm(FIGURE)

print (np.shape(assign_figure(FIGURE)))

ring1 = LinearRing(assign_figure(FIGURE))
x1, y1 = ring1.xy

ring2 = LinearRing(vertices)
x2, y2= ring2.xy

fig = pyplot.figure(1, figsize=(5,5), dpi=90)
example = fig.add_subplot(111)
example.plot(x1, y1, marker = 'o')
example.set_title(FIGURE)

ans = fig.add_subplot(111)
ans.plot(x2, y2)
ans.set_title(FIGURE)

    