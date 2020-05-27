import math
import itertools
DIST_T = 0.5


def vert_dist_is_valid(peak1, peak2):
    ksi11, ksi12, beta1, c_1 = [peak1["ksi1"], peak1["ksi2"],
                                peak1["beta"], peak1["acc value"]]
    ksi21, ksi22, beta2, c_2 = [peak2["ksi1"], peak2["ksi2"],
                                peak2["beta"], peak2["acc value"]]
    ang_dif = abs(beta1 - beta2)
    
    if ksi11 - ksi12 != 0 and ksi21 - ksi22 != 0:
        vert_dist_cond1 = abs((abs(ksi11 - ksi12) - c_1 * math.sin(ang_dif)) /
                           abs(ksi11 - ksi12))
        vert_dist_cond2 = abs((abs(ksi21 - ksi22) - c_2 * math.sin(ang_dif)) /
                           abs(ksi21 - ksi22))
        
        if max([vert_dist_cond1, vert_dist_cond2]) < DIST_T:
            return [True, ang_dif]

    return [False, ang_dif]


def gen_parallelograms_sides(peaks_list, hough_acc, img):
    parallelograms_sides = []
    
    for peak1, peak2 in itertools.combinations(peaks_list, 2):
        condition, ang_dif = vert_dist_is_valid(peak1, peak2)
        if condition:
            temp_dict = {"sides_a" : peak1,
                         "sides_b" : peak2,
                         "ang_dif" : ang_dif}
            parallelograms_sides.append(temp_dict)

    return parallelograms_sides
