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
