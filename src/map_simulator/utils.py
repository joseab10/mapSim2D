import numpy as np


def map_msg_to_numpy(msg):
    """
    Reshapes a map's data from a 1D list to a 2D ndarray.

    :param msg: (nav_msgs.OccupancyMap|gmapping.doubleMap) A map message.

    :return: (ndarray) The map, reshaped as a 2D matrix.
    """
    w = msg.info.width
    h = msg.info.height

    reshaped_map = np.array(msg.data)
    reshaped_map = reshaped_map.reshape(w, h)
    reshaped_map = np.flipud(reshaped_map)

    return reshaped_map


def map_msg_extent(msg):
    """
    Returns the extent of the map in world coordinates

    :param msg: ((nav_msgs.OccupancyMap|gmapping.doubleMap) A map message.

    :return: (list) The extents of the map in world coordinates [x0, x1, y0, y1]
    """

    w = msg.info.width
    h = msg.info.height

    # Set the plot's extension in world coordinates for meaningful plot ticks
    delta = msg.info.resolution
    x0 = msg.info.origin.position.x
    y0 = msg.info.origin.position.y
    x1 = x0 + w * delta
    y1 = y0 + h * delta

    extent = [x0, x1, y0, y1]

    return extent


def tf_frame_split(tf_frame):
    """
    Function for splitting a frame into its path components for easier comparison, ignoring slashes ('/').

    :param tf_frame: (string) TF frame

    :return: (list) List of TF Frame path components.
                    E.g.: for '/GT/base_link' it returns ['GT', 'base_link']
    """

    return filter(None, tf_frame.split('/'))


def tf_frame_join(*args):
    """
    Function for joining a frame list into a string path. Opposite to tf_frame_split.

    :param args: (string|list) Strings or List of strings for path components to be joined into a single TF Frame.

    :return: (string) A fully formed TF frame
    """

    tf_path = ''

    for arg in args:
        if isinstance(arg, list):
            tf_path += '/' + '/'.join(arg)
        elif isinstance(arg, str):
            tf_path += '/' + arg

    return tf_path[1:]


def tf_frame_normalize(tf_frame):
    """
    Function for normalizing a TF frame string.

    :param tf_frame: (string) String of a single TF Frame.

    :return: (string) A standardized TF frame
    """

    return tf_frame_join(tf_frame_split(tf_frame))


def tf_frame_eq(tf1, tf2):
    """
    Function for determining whether two TF chains are equal by ignoring slashes

    :param tf1: (string) First TF frame chain
    :param tf2: (string) Second TF frame chain

    :return: (bool) True if tf1 and tf2 represent the same path ignoring slashes
    """

    tf1_list = tf_frame_normalize(tf1)
    tf2_list = tf_frame_normalize(tf2)

    eq = tf1_list == tf2_list
    return eq


def world2map(point, map_origin, delta):
    """
    Convert from world units to discrete cell coordinates.

    :param point: (np.ndarray) X and Y position in world coordinates to be converted.
    :param map_origin: (np.ndarray) X and Y position in world coordinates of the map's (0, 0) cell.
    :param delta: (float) Width/height of a cell in world units (a.k.a. resolution).

    :return: (np.ndarray) Integer-valued coordinates in map units. I.e.: cell indexes corresponding to x and y.
    """

    int_point = point - map_origin
    int_point /= delta

    return int_point.astype(np.int)


def map2world(int_point, map_origin, delta, rounded=False):
    """
    Convert from discrete map cell coordinates to world units.

    :param int_point: (np.ndarray) Row and Column indices in map coordinates to be converted.
    :param map_origin: (np.ndarray) X and Y position in world coordinates of the map's (0, 0) cell.
    :param delta: (float) Width/height of a cell in world units (a.k.a. resolution).
    :param rounded: (bool)[Default: False] Round the resulting point up to an order of magnitude smaller
                                           than the resolution if True.
                                           Useful for repeatability when computing the center coordinates of cells.

    :return: (np.ndarray) X and Y position in world coordinates.
    """

    point = delta * np.ones_like(int_point)
    point = np.multiply(point, int_point)
    point += map_origin

    if rounded:
        decimals = np.log10(delta)
        if decimals < 0:
            decimals = int(np.ceil(-decimals) + 1)
            point = np.round(point, decimals)

    return point


def cell_centerpoint(point, map_origin, delta):
    """
    Gives the center point in world coordinates of the cell corresponding to a given point in world coordinates.

    :param point: (np.ndarray) X and Y position in world coordinates to be converted.
    :param map_origin: (np.ndarray) X and Y position in world coordinates of the map's (0, 0) cell.
    :param delta: (float) Width/height of a cell in world units (a.k.a. resolution).

    :return: (np.ndarray) X and Y position of the cell's center in world coordinates.
    """

    int_point = world2map(point, map_origin, delta)
    cnt_point = map2world(int_point, map_origin, delta, rounded=True)

    return cnt_point