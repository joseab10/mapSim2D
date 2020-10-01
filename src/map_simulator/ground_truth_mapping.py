# ROS Libraries
import rospy
import tf
import tf.transformations

# ROS Messages
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan

# Math Libraries
import numpy as np
from skimage.draw import line

from collections import defaultdict, deque

# Project Libraries
from map_simulator.utils import map2world, world2map


class GroundTruthMapping:
    """
    Class for generating a ground truth map from deterministic (noiseless) odometry and measurement messages,
    instead of running an entire SLAM stack.
    """

    def __init__(self):
        """
        Constructor
        """

        rospy.init_node('gt_mapping')

        self._tf_listener = tf.TransformListener()

        self._map_frame = rospy.get_param("~map_frame", "map")
        self._pose_origin_frame = rospy.get_param("~pose_origin_frame", "map")
        self._laser_frame = rospy.get_param("~sensor_frame", "/GT/laser_link")

        max_scan_buffer_len = rospy.get_param("~max_scan_buffer_len", 1000)
        self._occ_threshold = rospy.get_param("~occ_threshold", 0.25)

        self._sub_map = rospy.Subscriber("map", OccupancyGrid, self._map_callback)
        self._sub_scan = rospy.Subscriber("/GT/base_scan", LaserScan, self._sensor_callback)
        self._pub_map = rospy.Publisher("/GT/map", OccupancyGrid, queue_size=1)

        self._map_hits = defaultdict(int)
        self._map_visits = defaultdict(int)
        self._map_height = 0
        self._map_width = 0
        self._map_resolution = 0
        self._map_origin = None

        self._scan_buffer = deque(maxlen=max_scan_buffer_len)
        self._min_range = 0
        self._max_range = 0

        rospy.spin()

    def _sensor_callback(self, msg):
        """
        Function to be called each time a laser scan message is received.
        It computes the pose and endpoints of the laser beams and stores them in a queue,
        waiting for the right time to compute the map.

        :param msg: (sensor_msgs.LaserScan) Received Laser Scan message.

        :return: None
        """

        laser_pose = self._tf_listener.lookupTransform(self._pose_origin_frame, self._laser_frame, rospy.Time(0))
        lp = np.array([laser_pose[0][0], laser_pose[0][1]])  # Laser Pose
        _, _, lp_th = tf.transformations.euler_from_quaternion(laser_pose[1])  # Laser Orientation

        rospy.loginfo("Received scan at pose: ({}, {}), theta: {}".format(lp[0], lp[1], lp_th))
        self._min_range = msg.range_min
        self._max_range = msg.range_max

        ranges = np.array(msg.ranges)
        max_range = ranges > self._max_range
        ranges = np.clip(ranges, self._min_range, self._max_range)

        bearings = np.linspace(msg.angle_min, msg.angle_max, ranges.shape[0])
        bearings += lp_th

        cos_sin = np.stack([np.cos(bearings), np.sin(bearings)], axis=1)
        endpoints = np.multiply(ranges.reshape((-1, 1)), cos_sin)
        endpoints += lp

        self._scan_buffer.append((lp, endpoints, max_range))

    def _map_callback(self, msg):
        """
        Function to be called each time a map message is received,
        just to copy the SLAM generated map properties for easier comparison.
        It:
            * takes the metadata from the published map (height, width, resolution, map origin),
            * creates a map if it is the first time,
            * checks if the map size changed in subsequent times and enlarges it in case it did,
            * takes all the poses and endpoints from the queue and updates the map values,
            * thresholds the map values,
            * publishes a ground truth occupancy map

        :param msg: (nav_msgs.OccupancyGrid) Received Map message.

        :return: None
        """

        # Set map attributes
        self._map_height = msg.info.height
        self._map_width = msg.info.width
        self._map_resolution = msg.info.resolution
        self._map_origin = np.ndarray([msg.info.origin.position.x, msg.info.origin.position.y])

        # For each scan in the measurement list, convert the endpoints to the center points of the grid cells,
        # Get the cells crossed by the beams and mark those indexes as occ or free.
        while self._scan_buffer:
            scan = self._scan_buffer.popleft()
            lp = scan[0]
            ilp = world2map(lp, self._map_origin, self._map_resolution)

            endpoints = scan[1]
            max_range = scan[2]
            i_endpoints = world2map(endpoints, self._map_origin, self._map_resolution)
            for i, i_ep in enumerate(i_endpoints):

                line_cells = line(ilp[0], ilp[1], i_ep[0], i_ep[1])
                line_indexes = np.array(zip(line_cells[0], line_cells[1]))

                if not max_range[i]:
                    occ_indexes = line_indexes[-1]
                    # Increment hit cell
                    hit_c = tuple(map2world(occ_indexes, self._map_origin, self._map_resolution, rounded=True))
                    self._map_hits[hit_c] += 1

                # Increment visited cells
                for visit in line_indexes:
                    visit_c = tuple(map2world(visit, self._map_origin, self._map_resolution, rounded=True))
                    self._map_visits[visit_c] += 1

        # Compute Occupancy value as hits/visits from the default dicts
        map_shape = (self._map_width, self._map_height)
        tmp_map = -1 * np.ones(map_shape)
        for pos, visits in self._map_visits.iteritems():
            ix, iy = world2map(pos, self._map_origin, self._map_resolution)
            hits = self._map_hits[pos]
            tmp_map[ix, iy] = hits / visits

        tmp_occ = tmp_map >= self._occ_threshold
        tmp_free = np.logical_and(0 <= tmp_map, tmp_map < self._occ_threshold)
        occ_map = -1 * np.ones(map_shape, dtype=np.int8)
        occ_map[tmp_occ] = 100
        occ_map[tmp_free] = 0

        # Build Message and Publish
        map_header = Header()
        map_header.seq = msg.header.seq
        map_header.stamp = msg.header.stamp
        map_header.frame_id = self._map_frame

        map_info = MapMetaData()
        map_info.map_load_time = msg.info.map_load_time
        map_info.resolution = msg.info.resolution
        map_info.width = msg.info.width
        map_info.height = msg.info.height
        map_info.origin = msg.info.origin

        map_msg = OccupancyGrid()
        map_msg.header = map_header
        map_msg.info = map_info
        map_msg.data = np.ravel(occ_map.transpose()).tolist()

        self._pub_map.publish(map_msg)
