def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1))

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
# =============================================================================
#     ring = LinearRing(shape)
#     ring_shape = np.array(ring.coords)
#     diff_list = np.diff(ring_shape, axis=0)
#     idx = 0
#     for diff in diff_list:
#         if max(abs(diff[0]), abs(diff[1])) >= 3:
#             insert = getEquidistantPoints(ring_shape[idx], ring_shape[idx+1], 3)
#             insert = np.array(list(insert))
#             ring_shape = np.insert(ring_shape, idx+1, insert, axis=0)
#             idx += 4
#         else:
#             idx += 1
#     if np.amax(abs(ring_shape[-1]) - abs(ring_shape[0])) <= 1:
#         insert = (shape[-1] + shape[0]) / 2
#         ring_shape = np.add(ring_shape, insert)
#         
#     shape = ring_shape
# =============================================================================

            
    img = {"points": shape,
            "x_min": np.ceil(np.amin(shape[:, 0])),
            "x_max": np.ceil(np.amax(shape[:, 0])),
            "y_min": np.ceil(np.amin(shape[:, 1])),
            "y_max": np.ceil(np.amax(shape[:, 1])),
            "perimeter": Polygon(shape).length}       # 
    img["height"] = img["y_max"] - img["y_min"]
    img["width"] = img["x_max"] - img["x_min"]
    return img