'''
The algorithm which convert a noisy parallelogram 
into a set of four points - an ideal parallelogram.
'''

import numpy as np
import math
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
    elif figure_name == "example 2":
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
    return points

def find_max_points(points):
    """

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

def create_rho_theta_enh(x_max, y_max, rho_res):
    """
    Helper function for Hough Transform
    Creates rho and theta dimension for enchaced Hough accumulator

    Parameters
    ----------
    x_max : float
        The highest value by x axis
    y_max : float
        The highest value by y axis
    rho_res : float
        Rho resolution. Determines discretisation step for rho

    Returns
    -------
    theta : np.array
        Empty array representing theta dimension
    rho : np.array
        Empty array representing rho dimension

    """
     # Making theta dimension
    theta = np.arange(-90, 90, 180/(2*(y_max - 1)))
    # Making rho dimension:
    max_dist = np.sqrt((x_max - 1)**2 + (y_max - 1)**2)
    fac = np.ceil(max_dist/rho_res) # resolution factor
    rho = np.arange(-fac*rho_res, fac*rho_res, math.pi/4)
    return theta, rho

def create_rho_theta_normal(x_max, y_max, rho_res, theta_res):
    '''
    Helper function for Hough Transform
    Creates rho and theta dimension for not enchaced Hough accumulator

       Parameters
    ----------
    x_max : float
        The highest value by x axis
    y_max : float
        The highest value by y axis
    rho_res : float
        Rho resolution. Determines discretisation step for rho
    theta_res : float
        Theta resolution. Determines discretisation step for theta

    Returns
    -------
    theta : np.array
        Empty array representing theta dimension
    rho : np.array
        Empty array representing rho dimension

    '''
    theta = np.linspace(-90.0, 90.0, int(np.ceil(90.0/theta_res) + 1.0))
    #theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
    D = np.sqrt((x_max - 1)**2 + (y_max - 1)**2)
    q = np.ceil(D/rho_res)
    nrho = int(2*q + 1)
    rho = np.linspace(-q*rho_res, q*rho_res, nrho)
    return theta, rho

def fill_hs_acc(empty_acc, points, rho, theta):
    """
    Compute voices for every (rho, theta) pair in Hough Accumulator, and update it values

    Parameters
    ----------
    empty_acc : np.array
        Empty array [rho x theta]
    points : np.array
        2D array of points to transform
    rho : float
        Ordered array of rhos represented by rows in C
    theta : np.array
        Ordered array of thetas represented by columns in C

    Returns
    -------
    None.

    """
    for x, y in points:
        for thIdx in range(len(theta)):
            rhoVal = x*math.cos(theta[thIdx]*math.pi/180.0) + \
                    y*math.sin(theta[thIdx]*math.pi/180.0)
            rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
            empty_acc[rhoIdx, thIdx] += 1

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
    C_enh : np.array
        Enhansed version of C used to extract highest peaks more easily

    """
    h = len(rho)
    w = len(theta)
    C_integr = np.sum(ht_acc[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)])
    C_enh = h*w*(ht_acc**2)/C_integr
    return C_enh

def hough_transform(points, enhanced, theta_res=1.0, rho_res=1.0):
    """
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
    if enhanced:
        x_max, y_max = find_max_points(points)
        theta, rho = create_rho_theta_enh(x_max, y_max, rho_res)
    else:
        x_max, y_max = find_max_points(points)
        theta, rho = create_rho_theta_normal(x_max, y_max, rho_res, theta_res)
    C = np.zeros((len(rho), len(theta)))
    fill_hs_acc(C, points, rho, theta)
    theta_T = 3 * 180/(2*(y_max - 1)) # Create a theta threshold
    if enhanced:
        C = enhance_hs_acc(C, rho, theta)
    return theta, rho, C, theta_T

def top_n_rho_theta_pairs(ht_acc, n, rhos, thetas):
    '''
    

    Parameters
    ----------
    ht_acc : np.array
        Hough transform accumulator matrix C (rho by theta)
    n : int
        n pairs of rho and thetas desired
    rhos : list
        Ordered array of rhos represented by rows in C
    thetas : list
        Ordered array of thetas represented by columns in C

    Returns
    -------
    list
        Top n rho theta pairs in H by accumulator value
        '''
    
    flat = list(set(np.hstack(ht_acc)))
    flat_sorted = sorted(flat, key = lambda n: -n) 
    coords_sorted = [(np.argwhere(ht_acc == acc_value)) for acc_value in flat_sorted[0:n]]
    rho_theta = []
    x_y = [] 
    for coords_for_val_idx in range(0, len(coords_sorted), 1):
      coords_for_val = coords_sorted[coords_for_val_idx]
      for idx in range(0, len(coords_for_val), 1):
        k,m = coords_for_val[idx] # k by m matrix
        rho = rhos[k]
        theta = thetas[m]
        rho_theta.append([rho, theta])
        x_y.append([k, n]) # just to unnest and reorder coords_sorted
    return [rho_theta[0:n], x_y]



def peaks_to_dict(peak_pairs):
    peaks_dict_list = []
    temp_dict = {"rhos": None, "thetas": None, "acc_values": None}
    for pair in peak_pairs:
        temp_dict = {"rhos": [pair[0][0], pair[2][0]], 
                     "thetas": [pair[0][1], pair[2][1]], 
                     "acc_values": [pair[1], pair[3]]}
        peaks_dict_list.append(temp_dict)
    return peaks_dict_list

def choose_peaks(ht_acc, peaks, rhos, thetas, theta_T, len_T = 0.5):
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
    satisfying_pairs = []
    for current_peak in peaks[:-1]:
        cur_idx = peaks.index(current_peak)
        for compare_peak in peaks[cur_idx + 1:]:
            rho1, theta1 = current_peak
            rho2, theta2 = compare_peak
            acc_idx1 = [np.where(rhos == rho1), np.where(thetas == theta1)]
            acc_idx2 = [np.where(rhos == rho2), np.where(thetas == theta2)]
            acc_value1 = ht_acc[acc_idx1[0], acc_idx1[1]]
            acc_value2 = ht_acc[acc_idx2[0], acc_idx2[1]]
            is_parallel = abs(theta1 - theta2) < theta_T
            is_apropriate_dist = abs(acc_value1- acc_value2) < len_T * (acc_value1 + acc_value2) / 2
            if is_parallel and is_apropriate_dist:
                satisfying_pairs.append([current_peak, float(acc_value1), compare_peak, float(acc_value2)])
    return peaks_to_dict(satisfying_pairs)



def gen_extended_peaks(peak_pairs):
    """
    

    Parameters
    ----------
    pairs : list
        List of chosen pairs of peaks from Hough Accumulator and their parameters.
        List must be given in form: [[rho1, theta1], acc_value1, [rho2, theta2], acc_value2]

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
    ang_dif = abs(beta1 - beta2) * math.pi / 180.0
    vert_dist_cond1 = abs(ksi11 - ksi12) == C1 * math.sin(ang_dif)
    vert_dist_cond2 = abs(ksi21 - ksi22) == C2 * math.sin(ang_dif)
    if vert_dist_cond1 and vert_dist_cond2:
        print (ksi11 - ksi12)
        print ("Differense 1: ", abs(ksi11 - ksi12) - C1 * math.sin(ang_dif))
        print ("Differense 2: ", abs(ksi21 - ksi22) - C2 * math.sin(ang_dif))
        return [True, ang_dif]
    else:
        print (ksi11 - ksi12)
        print ("Differense 1: ", abs(ksi11 - ksi12) - C1 * math.sin(ang_dif))
        print ("Differense 2: ", abs(ksi21 - ksi22) - C2 * math.sin(ang_dif))
    return [False, ang_dif]

def find_valid_peaks_pair(peaks):
    for current_peak in peaks[:-1]:
        cur_idx = peaks.index(current_peak)
        for other_peak in peaks[cur_idx + 1:]:
            if vert_dist_is_valid(current_peak, other_peak)[0]:
                return [current_peak, other_peak]
            
            

def run_algorithm(figure, enh = True):
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
    thetas, rhos, C, theta_T = hough_transform(figure, enh)
    rho_theta_pairs, x_y_pairs= top_n_rho_theta_pairs(C, 8, rhos, thetas)
    satisfying_pairs = choose_peaks(C, rho_theta_pairs, rhos, thetas, theta_T)
    extended_peaks = gen_extended_peaks(satisfying_pairs)
    valid_peaks = find_valid_peaks_pair(extended_peaks)
    return thetas, rhos, C, rho_theta_pairs, x_y_pairs, satisfying_pairs, extended_peaks, valid_peaks

#================================================TEST CASES=======================================================

#Square
thetas, rhos, C_ench, rho_theta_pairs, x_y_pairs, satisfying_pairs, extended_peaks, valid_peaks = run_algorithm("square", False)


