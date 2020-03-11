import numpy as np
import math
edges1 = np.array([[8.04411, 78.9279], 
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

edges2 = np.array([[17.5544, 72.9043],
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

edges3 = np.array([[9.45637, 65.319],
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

line = np.array([[1,1],
         [2,2],
         [3,3],
         [4,4]])
line = line * 10

square = np.array([[10,10],
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

def hough_transform(points, theta_res=1, rho_res=1):
    '''..
    @param points array to transform
    @param resolution of theta
    @param resolution of rho
    @return Hough Accumulator matrix
    '''
    x_max = np.ceil(np.amax(points[:, 0]))
    y_max = np.ceil(np.amax(points[:, 1]))
    # Making theta dimension:
    #theta = np.linspace(-90, 90, np.ceil(180.0/theta_res) + 1.0)
    theta = np.arange(-100, 100, 180/(2*(y_max - 1)))
    # Making rho dimension:
    D = np.sqrt((x_max+10)**2 + (y_max+10)**2)
    q = np.ceil(D/rho_res)
    rho = np.arange(-q*rho_res, q*rho_res, math.pi/4)
    '''
    nrho = 2*q + 1
    rho = np.linspace(-q*rho_res, q*rho_res, nrho)
    '''
    # Initialize an empty Hough Accumulator:
    H = np.zeros((len(rho), len(theta)))
    
    # Making Hough Space:
    for x, y in points:
        for thIdx in range(len(theta)):
            rhoVal = x*math.cos(theta[thIdx]*math.pi/180.0) + \
                    y*math.sin(theta[thIdx]*math.pi/180.0)
            rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
            H[rhoIdx[0], thIdx] += 1
    # Enchanced accumulator model
    h = len(rho)
    w = len(theta)
    H_integr = np.sum(H[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)])
    H_ench = h*w*(H**2)/H_integr

    return theta, rho, H, H_ench

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
      k,m = coords_for_val[i] # k by m matrix
      rho = rhos[k]
      theta = thetas[m]
      rho_theta.append([rho, theta])
      x_y.append([k, n]) # just to unnest and reorder coords_sorted
  return [rho_theta[0:n], x_y]

thetas1, rhos1, H1, H1_ench = hough_transform(edges1, 10, 10)
thetas2, rhos2, H2, H2_ench = hough_transform(edges2, 1, 1)
thetas3, rhos3, H3, H3_ench = hough_transform(edges3, 1, 1)
    
rho_theta_pairs1, x_y_pairs1 = top_n_rho_theta_pairs(H1_ench, 4, rhos1, thetas1)
rho_theta_pairs2, x_y_pairs2 = top_n_rho_theta_pairs(H2_ench, 4, rhos2, thetas2)
rho_theta_pairs3, x_y_pairs3 = top_n_rho_theta_pairs(H3_ench, 100, rhos3, thetas3)


# Samples check
#thetasl, rhosl, Hl = hough_transform(line)
thetassq, rhossq, Hsq, Hsq_ench = hough_transform(square, 1, 1)
rho_theta_pairssq, x_y_pairssq = top_n_rho_theta_pairs(Hsq, 4, rhossq, thetassq)