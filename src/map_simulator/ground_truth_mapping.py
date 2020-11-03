# ROS Libraries
import rospy
import tf
import tf2_ros
import tf.transformations

# ROS Messages
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

# Math Libraries
import numpy as np
from skimage.draw import line

from collections import defaultdict, deque

# Project Libraries
from map_simulator.utils import map2world, world2map, tf_frame_normalize


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

        self._map_frame = tf_frame_normalize(rospy.get_param("~map_frame", "map"))

        max_scan_buffer_len = rospy.get_param("~max_scan_buffer_len", 1000)
        self._occ_threshold = rospy.get_param("~occ_threshold", 0.25)

        self._sub_map = rospy.Subscriber("/map", OccupancyGrid, self._map_callback)
        self._sub_scan = rospy.Subscriber("/GT/base_scan", LaserScan, self._sensor_callback)
        self._sub_doLoc = rospy.Subscriber("doLocOnly", Bool, self._loc_only_callback)
        self._pub_map = rospy.Publisher("/GT/map", OccupancyGrid, queue_size=1)

        self._map_resolution = None

        self._scan_buffer = deque(maxlen=max_scan_buffer_len)

        self._map_hits = defaultdict(int)
        self._map_visits = defaultdict(int)

        self._loc_only = False

        rospy.spin()

    def _loc_only_callback(self, msg):
        self._loc_only = msg.data

    def _sensor_callback(self, msg):
        """
        Function to be called each time a laser scan message is received.
        It computes the pose and endpoints of the laser beams and stores them in a queue,
        waiting for the right time to compute the map.

        :param msg: (sensor_msgs.LaserScan) Received Laser Scan message.

        :return: None
        """

        # Stop registering scans if no more mapping is taking place
        if self._loc_only:
            return

        try:
            self._tf_listener.waitForTransform(self._map_frame, msg.header.frame_id, msg.header.stamp,
                                               rospy.Duration(2))
            laser_pose, laser_orientation = self._tf_listener.lookupTransform(self._map_frame, msg.header.frame_id,
                                                                              msg.header.stamp)

        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException, tf2_ros.TransformException) as e:
            rospy.logwarn("Couldn't find transform for scan {}. {}".format(msg.header.seq, e))
            return

        lp = np.array([laser_pose[0], laser_pose[1]])  # Laser Pose
        _, _, lp_th = tf.transformations.euler_from_quaternion(laser_orientation)  # Laser Orientation

        self._min_range = msg.range_min
        self._max_range = msg.range_max

        ranges = np.array(msg.ranges)
        max_range = ranges >= self._max_range
        ranges = np.clip(ranges, self._min_range, self._max_range)

        bearings = np.linspace(msg.angle_min, msg.angle_max, ranges.shape[0])
        bearings += lp_th

        cos_sin = np.stack([np.cos(bearings), np.sin(bearings)], axis=1)
        endpoints = np.multiply(ranges.reshape((-1, 1)), cos_sin)
        endpoints += lp

        if self._map_resolution is None:
            # Store the scan data until we receive our first map message and thus know the map's resolution
            self._scan_buffer.append((lp, endpoints, max_range))
        else:
            self._register_scan(lp, endpoints, max_range)

        rospy.loginfo("Scan {} received and added to buffer at pose ({}): ({:.3f}, {:.3f}), theta: {:.3f}.".format(
            msg.header.seq, msg.header.frame_id, lp[0], lp[1], lp_th))

    def _register_scan(self, laser_pose, endpoints, max_range):
        ilp = world2map(laser_pose, np.zeros(2), self._map_resolution)
        i_endpoints = world2map(endpoints, np.zeros(2), self._map_resolution)

        for i, i_ep in enumerate(i_endpoints):

            line_rows, line_cols = line(ilp[0], ilp[1], i_ep[0], i_ep[1])
            line_indexes = np.array(zip(line_rows, line_cols))

            if not max_range[i]:
                occ_indexes = line_indexes[-1]
                # Increment hit cell
                hit_c = tuple(map2world(occ_indexes, np.zeros(2), self._map_resolution, rounded=True))
                self._map_hits[hit_c] += 1

            # Increment visited cells
            for visit in line_indexes:
                visit_c = tuple(map2world(visit, np.zeros(2), self._map_resolution, rounded=True))
                self._map_visits[visit_c] += 1

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
        height = msg.info.height
        width = msg.info.width
        if self._map_resolution is None:
            self._map_resolution = msg.info.resolution

        if msg.info.resolution != self._map_resolution:
            raise ValueError("Map resolution changed from last time {}->{}. I can't work in these conditions!".format(
                self._map_resolution, msg.info.resolution))

        map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])

        # For each scan in the measurement list, convert the endpoints to the center points of the grid cells,
        # Get the cells crossed by the beams and mark those indexes as occ or free.
        while self._scan_buffer:
            laser_pose, endpoints, max_range = self._scan_buffer.popleft()
            self._register_scan(laser_pose, endpoints, max_range)

        # Compute Occupancy value as hits/visits from the default dicts
        map_shape = (width, height)
        occ_map = -1 * np.ones(map_shape, dtype=np.int8)

        # Freeze a snapshot (copy) of the current hits and visits in case a scan message comes in and alters the values.
        visits_snapshot = self._map_visits.copy()
        hits_snapshot = self._map_hits.copy()
        for pos, visits in visits_snapshot.iteritems():
            if visits <= 0:
                continue

            ix, iy = world2map(pos, map_origin , self._map_resolution)
            # Ignore cells not contained in image
            if ix > width or iy > height:
                continue

            hits = hits_snapshot[pos]
            tmp_occ = float(hits) / float(visits)
            if 0 <= tmp_occ < self._occ_threshold:
                occ_map[ix, iy] = 0
            elif tmp_occ >= self._occ_threshold:
                occ_map[ix, iy] = 100

        # Build Message and Publish
        map_msg = OccupancyGrid()
        map_msg.header = msg.header
        map_msg.info = msg.info
        map_msg.data = np.ravel(np.transpose(occ_map)).tolist()

        rospy.loginfo("Publishing map at {} with seq {}.".format(self._map_frame, msg.header.seq))

        self._pub_map.publish(map_msg)
