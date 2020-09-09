# ROS Libraries
import rospy
import roslib
import tf

# ROS Messages
from std_msgs.msg import Int8MultiArray, Header
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan

# Math Libraries
import numpy as np
from skimage.draw import line


class GroundTruthMapping:
    def __init__(self):

        rospy.init_node('gt_mapping')

        self._tf_listener = tf.TransformListener()

        self._gt_prefix = rospy.get_param("~gt_prefix", "/GT/")

        self._map_frame = rospy.get_param("~map_frame", "map")
        self._laser_frame = rospy.get_param("~sensor_frame", self._gt_prefix + "laser_link")

        self._max_scan_buffer_len = rospy.get_param("~max_scan_buffer_len", 1000)
        self._occ_threshold = rospy.get_param("~occ_threshold", 0.25)

        self._sub_map = rospy.Subscriber("map", OccupancyGrid, self._map_callback)
        self._sub_scan = rospy.Subscriber(self._gt_prefix + "base_scan", LaserScan, self._sensor_callback)
        self._pub_map = rospy.Publisher(self._gt_prefix + "map", OccupancyGrid, queue_size=1)

        self._map_hits = None
        self._map_visits = None
        self._map_height = 0
        self._map_width = 0
        self._map_resolution = 0
        self._map_origin = None

        self._endpoints = []
        self._min_range = 0
        self._max_range = 0

        rospy.spin()

    def _world2map(self, x, y, mx0=None, my0=None, delta=None):

        if mx0 is None:
            mx0 = self._map_origin.position.x
        if my0 is None:
            my0 = self._map_origin.position.y
        if delta is None:
            delta = self._map_resolution

        ix = int((x - mx0) // delta)
        iy = int((y - my0) // delta)

        return ix, iy

    def _sensor_callback(self, msg):
        pose = self._tf_listener.lookupTransform(self._map_frame, self._laser_frame, rospy.Time(0))

        self._min_range = msg.range_min
        self._max_range = msg.range_max

        _, _, yaw = tf.transformations.euler_from_quaternion(pose[1])
        pose_x = pose[0][0]
        pose_y = pose[0][1]

        ranges = np.array(msg.ranges)

        bearings = yaw + np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        bearings = np.append(bearings, msg.angle_max)

        meas_x = np.multiply(ranges, np.cos(bearings))
        meas_x += pose_x
        meas_y = np.multiply(ranges, np.sin(bearings))
        meas_y += pose_y

        endpoints = np.stack([meas_x, meas_y], axis=1)

        self._endpoints.append(((pose_x, pose_y), endpoints))

    def _map_callback(self, msg):
        # Get published map properties
        height = msg.info.height
        width = msg.info.width
        resolution = msg.info.resolution
        origin = msg.info.origin

        # If this is the first time, create a new blank map
        if self._map_hits is None:
            self._map_height = height
            self._map_width = width
            self._map_resolution = resolution
            self._map_origin = origin

            self._map_hits = np.zeros((width, height), dtype=np.int)
            self._map_visits = np.zeros((width, height), dtype=np.int)

        # If the size changed, increase the size of the map
        if self._map_height < height or self._map_width < width:
            # Create new map with new dimensions
            new_map_hits = np.zeros((width, height), dtype=np.int)
            new_map_visits = np.zeros((width, height), dtype=np.int)

            # Compute grid position of old map's origin
            x0, y0 = self._world2map(self._map_origin.position.x, self._map_origin.position.y,
                                     mx0=origin.position.x, my0=origin.position.y, delta=resolution)
            # Compute grid position of old map's opposite corner
            x1 = x0 + self._map_width
            y1 = y0 + self._map_height
            # Copy data from the old map to the new one
            new_map_hits[x0:x1, y0:y1] = self._map_hits
            new_map_visits[x0:x1, y0:y1] = self._map_visits

            # Set map attributes
            self._map_height = height
            self._map_width = width
            self._map_resolution = resolution
            self._map_origin = origin
            # Set maps to new object
            self._map_hits = new_map_hits
            self._map_visits = new_map_visits

        # For each scan in the measurement list, convert the endpoints to grid coordinates,
        # Get the cells crossed by the lines and mark those indexes as occ or free.
        while self._endpoints:
            scan = self._endpoints.pop()
            pose = scan[0]
            ix0, iy0 = self._world2map(pose[0], pose[1])

            measurements = scan[1]
            for endpoint in measurements:
                ix1, iy1 = self._world2map(endpoint[0], endpoint[1])

                line_cells = line(ix0, iy0, ix1, iy1)
                line_indexes = np.array(zip(line_cells[0], line_cells[1]))
                occ_indexes = line_indexes[-1]

                self._map_hits[occ_indexes[0], occ_indexes[1]] += 1
                self._map_visits[line_cells[0], line_cells[1]] += 1

        # Compute occupancy
        tmp_map = np.divide(self._map_hits, self._map_visits,
                            out=-1 * np.ones_like(self._map_hits), where=self._map_visits != 0)
        tmp_occ = tmp_map >= self._occ_threshold
        tmp_free = np.logical_and(0 <= tmp_map, tmp_map < self._occ_threshold)
        occ_map = -1 * np.ones((self._map_width, self._map_height), dtype=np.int8)
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
