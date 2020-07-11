# ROS Libraries
import rospy
import roslib
import tf
import rosbag

# ROS Messages
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

import os.path
from time import sleep, time

import simplejson as json

# Math Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# Project Libraries
from geometry import Line, Polygon, rotate2d


def set_dict_param(in_dict, self_dict, key, param_name, default):
    if key in in_dict:
        self_dict[key] = in_dict[key]
    else:
        rospy.logwarn("No {} defined in config file. Using default value: '{}'.".format(param_name, default))
        self_dict[key] = default


class MapSimulator2D:

    def __init__(self, map_file, robot_file):
        map_config = {}
        in_path = ""

        try:
            in_path = os.path.expanduser(map_file)
            with open(in_path, "r") as f:
                text = f.read()
                map_config = json.loads(text)

        except (IOError, ValueError):
            rospy.logerr("Couldn't open %", map_file)
            exit(-1)

        rospy.loginfo("Reading data from : %s", in_path)

        # Map Obstacle Parsing
        obstacles = []

        try:
            config_obstacles = map_config['obstacles']

        except KeyError:
            rospy.logwarn("No obstacles defined in map file")
            config_obstacles = []

        for obstacle in config_obstacles:
            try:
                obstacle_type = obstacle['type']
            except KeyError:
                rospy.logwarn("Obstacle has no type, ignoring")
                continue

            if obstacle_type == "polygon":
                try:
                    vertices = obstacle['vertices']
                    obstacles.append(Polygon(vertices))
                except KeyError:
                    rospy.logwarn("Polygon Obstacle has no vertices defined")

            # TODO: Define more types of obstacles
            elif obstacle_type == "circle":
                pass
            elif obstacle_type == "bezier_pline":
                pass

        self._obstacles = obstacles

        config = {}

        try:
            in_path = os.path.expanduser(robot_file)
            with open(in_path, "r") as f:
                text = f.read()
                config = json.loads(text)

        except (IOError, ValueError):
            rospy.logerr("Couldn't open %", robot_file)
            exit(-1)

        rospy.loginfo("Reading data from : %s", in_path)

        self._params = {}

        # Transform Parameters
        set_dict_param(config, self._params, 'odom_frame', 'Odometry TF Frame', 'odom')
        set_dict_param(config, self._params, 'base_frame', 'Base TF Frame', 'base_link')
        set_dict_param(config, self._params, 'laser_frame', 'Laser TF Frame', 'base_link')

        set_dict_param(config, self._params, 'odom_to_base_tf', 'Odometry to Base Transform', [[0, 0], 0])
        set_dict_param(config, self._params, 'base_to_laser_tf', 'Base to Laser Transform', [[0, 0], 0])

        set_dict_param(config, self._params, 'scan_topic', 'ROS Topic for scan messages', 'base_scan')

        # Uncertainty Parameters
        set_dict_param(config, self._params, 'deterministic', 'Deterministic process', False)

        set_dict_param(config, self._params, 'odometry_sigma', 'Odometry Covariance Matrix', [[0, 0, 0],
                                                                                              [0, 0, 0],
                                                                                              [0, 0, 0]])
        self._params['odometry_sigma'] = np.array(self._params['odometry_sigma'])

        set_dict_param(config, self._params, 'measurement_sigma', 'Measurement Covariance Matrix', [[0, 0],
                                                                                                    [0, 0]])
        self._params['measurement_sigma'] = np.array(self._params['measurement_sigma'])

        # Scan Parameters
        set_dict_param(config, self._params, 'num_rays', 'Number of Sensor beams per scan', 180)
        set_dict_param(config, self._params, 'start_ray', 'Starting scan angle (in Radians)', - np.pi / 2)
        set_dict_param(config, self._params, 'end_ray', 'End scan angle (in Radians)', - np.pi / 2)
        set_dict_param(config, self._params, 'max_range', 'Laser Scanner Max. Range (in m)', 20)

        set_dict_param(config, self._params, 'meas_per_move', 'Number of measurements per move command', 5)

        # Timestamp parameters
        set_dict_param(config, self._params, 'initial_timestamp', 'Initial Time Stamp', None)
        if self._params['initial_timestamp'] is None:
            self._current_time = rospy.Time.from_sec(time())
        else:
            self._current_time = rospy.Time(int(self._params['initial_timestamp']))
        set_dict_param(config, self._params, 'move_time_interval', 'Time in ms that a movement command takes', 1000.0)
        set_dict_param(config, self._params, 'scan_time_interval', 'Time (in ms) that a measurement command takes', 50.0)

        self._moves = []

        # Message Counters
        self._laser_msg_seq = 0
        self._tf_msg_seq = 0

        try:
            self._moves = config['move_commands']
        except KeyError:
            rospy.logwarn("No moves defined in config file. Considering only starting position")

        self._position = np.zeros(2)
        self._orientation = np.zeros(1)
        self._sensor_position = np.zeros(2)
        self._sensor_orientation = np.zeros(1)

        try:
            self._position = np.array(config['start_pose'][0])
        except KeyError:
            rospy.logwarn("No initial position defined in config file. Starting at (0, 0)")

        try:
            self._orientation = np.array(config['start_pose'][1])
        except KeyError:
            rospy.logwarn("No initial orientation defined in config file. Starting with theta=0")

        self._compute_sensor_pose()

    def convert(self, output_file, display=False):

        bag = None
        out_path = ""

        try:
            out_path = os.path.expanduser(output_file)
            bag = rosbag.Bag(out_path, "w")

        except (IOError, ValueError):
            rospy.logerr("Couldn't open %", output_file)
            exit(-1)

        rospy.loginfo("Writing rosbag to : %s", out_path)

        axes = None

        self._add_tf_msg(bag, update_laser_tf=True)

        if display:
            plt.ion()
            figure1 = plt.figure(1)
            axes = figure1.add_subplot(111)
            figure1.show()

            self._render(axes)

        for move in self._moves:

            self._move(move)
            self._add_tf_msg(bag)

            if display:
                self._render(axes)

            measurements, endpoints, hits = self._ray_trace()

            for i in range(int(self._params['meas_per_move'])):
                if self._params['deterministic']:
                    noisy_meas = measurements
                    meas_noise = np.zeros(2)
                else:
                    meas_noise = np.random.multivariate_normal(np.zeros(2), self._params['measurement_sigma'],
                                                               size=measurements.shape[0])
                    noisy_meas = measurements + meas_noise

                self._add_scan_msg(bag, noisy_meas)

                if display:
                    if self._params['deterministic']:
                        noisy_endpoints = endpoints
                    else:
                        range_noises = meas_noise[:, 1]
                        bearing_noises = meas_noise[:, 0]
                        bearing_noises = np.array([np.cos(bearing_noises), np.sin(bearing_noises)])
                        endpoint_noise = range_noises * bearing_noises
                        endpoint_noise = endpoint_noise.transpose()
                        noisy_endpoints = endpoints + endpoint_noise

                    self._render(axes, noisy_endpoints, hits)

        bag.close()

        rospy.loginfo("Finished simulation and saved to rosbag")

    def _move(self, move_cmd):
        old_pos = np.concatenate((self._position, self._orientation))

        if move_cmd['type'] == 'pose':
            if self._params['deterministic']:
                noise = np.zeros_like(old_pos)
            else:
                noise = np.random.multivariate_normal(np.zeros_like(old_pos), self._params['odometry_sigma'])

            self._position = np.array(move_cmd['params'][0] + noise[0:1]).flatten()
            self._orientation = np.array(move_cmd['params'][1] + noise[2]).flatten()

        elif move_cmd['type'] == "odom":
            # TODO
            pass
        elif move_cmd['type'] == "velocity":
            # TODO
            pass

        # Recompute sensor pose from new robot pose
        self._compute_sensor_pose()

    def _compute_sensor_pose(self):
        tf_trans = np.array(self._params['base_to_laser_tf'][0])
        tf_rot = np.array(self._params['base_to_laser_tf'][1])

        rotation = rotate2d(self._orientation)
        translation = rotation.dot(tf_trans)

        self._sensor_position = self._position + translation
        self._sensor_orientation = self._orientation + tf_rot

    def _ray_trace(self):

        bearing_ranges = []
        endpoints = []
        hits = []

        bearing = self._params['start_ray']

        num_rays = self._params['num_rays']
        if num_rays > 1:
            num_rays -= 1

        bearing_increment = (self._params['end_ray'] - self._params['start_ray']) / num_rays

        for i in range(int(self._params['num_rays'])):

            theta = self._sensor_orientation + bearing
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([c, s]).flatten()
            max_ray_endpt = self._sensor_position + self._params['max_range'] * rotation_matrix
            ray = Line(self._sensor_position, max_ray_endpt)

            min_range = self._params['max_range']
            min_endpt = max_ray_endpt
            hit = False

            for obstacle in self._obstacles:

                intersect = obstacle.line_intersects(ray)

                if intersect is not None:
                    beam = Line(ray.p1, intersect)
                    meas_range = beam.len
                    if min_range is None or meas_range < min_range:
                        min_range = meas_range
                        min_endpt = intersect
                        hit = True

            bearing_ranges.append([bearing, min_range])
            endpoints.append(min_endpt)
            hits.append(hit)

            bearing += bearing_increment

        bearing_ranges = np.array(bearing_ranges)
        endpoints = np.array(endpoints)
        hits = np.array(hits)

        return bearing_ranges, endpoints, hits

    def _add_tf_msg(self, bag, update_laser_tf=True):
        tf_odom_robot_msg = TransformStamped()

        tf2_msg = TFMessage()

        tf_odom_robot_msg.header.stamp = self._current_time
        tf_odom_robot_msg.header.seq = self._tf_msg_seq
        tf_odom_robot_msg.header.frame_id = self._params['odom_frame']
        tf_odom_robot_msg.child_frame_id = self._params['base_frame']

        position = Point(float(self._position[0]), float(self._position[1]), 0.0)
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, float(self._orientation))

        tf_odom_robot_msg.transform.translation = position
        tf_odom_robot_msg.transform.rotation.x = quaternion[0]
        tf_odom_robot_msg.transform.rotation.y = quaternion[1]
        tf_odom_robot_msg.transform.rotation.z = quaternion[2]
        tf_odom_robot_msg.transform.rotation.w = quaternion[3]

        tf2_msg.transforms.append(tf_odom_robot_msg)

        if update_laser_tf:
            tf_laser_robot_msg = TransformStamped()

            tf_laser_robot_msg.header.stamp = self._current_time
            tf_laser_robot_msg.header.seq = self._tf_msg_seq
            tf_laser_robot_msg.header.frame_id = self._params['base_frame']
            tf_laser_robot_msg.child_frame_id = self._params['laser_frame']

            position = Point(float(self._params['base_to_laser_tf'][0][0]),
                             float(self._params['base_to_laser_tf'][0][1]), 0.0)
            quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, float(self._params['base_to_laser_tf'][1][0]))

            tf_laser_robot_msg.transform.translation = position
            tf_laser_robot_msg.transform.rotation.x = quaternion[0]
            tf_laser_robot_msg.transform.rotation.y = quaternion[1]
            tf_laser_robot_msg.transform.rotation.z = quaternion[2]
            tf_laser_robot_msg.transform.rotation.w = quaternion[3]

            tf2_msg.transforms.append(tf_laser_robot_msg)

        bag.write("/tf", tf2_msg, self._current_time)

        self._tf_msg_seq += 1
        secs = self._params['move_time_interval']
        nsecs = secs
        secs = int(secs / 1000)
        nsecs = int((nsecs - 1000 * secs) * 1000)
        self._current_time += rospy.Duration(secs, nsecs)

    def _add_scan_msg(self, bag, measurements):

        meas_msg = LaserScan()

        meas_msg.header.frame_id = self._params['laser_frame']
        meas_msg.header.stamp = self._current_time
        meas_msg.header.seq = self._laser_msg_seq

        meas_msg.angle_min = self._params['start_ray']
        meas_msg.angle_max = self._params['end_ray']
        if self._params['num_rays'] > 1:
            meas_msg.angle_increment = (meas_msg.angle_max - meas_msg.angle_min) / (self._params['num_rays'] - 1)
        else:
            meas_msg.angle_increment = 0.0

        meas_msg.range_min = 0.0
        meas_msg.range_max = self._params['max_range']

        meas_msg.ranges = measurements[:, 1]
        meas_msg.intensities = []

        bag.write(self._params['scan_topic'], meas_msg, meas_msg.header.stamp)

        self._laser_msg_seq += 1
        secs = self._params['scan_time_interval']
        nsecs = secs
        secs = int(secs/1000)
        nsecs = int((nsecs - 1000 * secs) * 1000)
        self._current_time += rospy.Duration(secs, nsecs)

    def _draw_map(self, ax):
        for obstacle in self._obstacles:
            if isinstance(obstacle, Polygon):
                vertices = obstacle.vertices.transpose()
                ax.fill(vertices[0], vertices[1], edgecolor='tab:blue', hatch='////', fill=False)

    def _draw_robot(self, ax):
        robot_size = 0.05

        robot_base = plt.Circle((self._position[0], self._position[1]), robot_size, color='tab:green', zorder=2)
        ax.add_artist(robot_base,)

        orientation_inipt = self._position
        orientation_endpt = robot_size * np.array([np.cos(self._orientation), np.sin(self._orientation)]).reshape((2,))
        orientation_endpt += orientation_inipt
        orientation_line = np.stack((orientation_inipt, orientation_endpt)).transpose()

        robot_orientation = plt.Line2D(orientation_line[0], orientation_line[1], color='white')
        ax.add_artist(robot_orientation)

    def _draw_beams(self, ax, beams, hits):
        if beams is None:
            return

        for i, beam in enumerate(beams):
            ray = np.array([self._sensor_position, beam])
            ray = ray.transpose()

            if hits[i]:
                ax.plot(ray[0], ray[1], 'tab:red', marker='.', linewidth=0.5)
            else:
                ax.plot(ray[0], ray[1], 'tab:red', dashes=[4, 1], linewidth=0.5)

    def _render(self, ax, beam_endpoints=None, hits=None):
        """
        Renders a graphical view of the map, the current state of the robot and the measurements using Matplotlib

        :param ax: Axes object so as to not redefine and recreate it every time a frame is rendered
        :param beam_endpoints: Numpy array of the laser endpoints. None if measurements are not to be displayed.
        :param hits: Numpy array of booleans indicating whether each laser beam actually hit an obstacle or was a
        max. range reading.

        :return: None
        """

        ax.clear()

        self._draw_map(ax)
        self._draw_beams(ax, beam_endpoints, hits)
        self._draw_robot(ax)

        ax.set_aspect('equal', 'datalim')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#CCCCCC')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')

        ax.grid(True)

        plt.draw()
        plt.pause(0.0001)

        sleep(0.5)


if __name__ == '__main__':
    import argparse

    rospy.init_node('map_simulator', anonymous=True)

    parser = argparse.ArgumentParser(description="Generate a ROSbag file from a simulated robot trajectory.")

    parser.add_argument('map_file', action='store', help='Input JSON map file', type=str)
    parser.add_argument('robot_file', action='store', help='Input JSON robot config file', type=str)
    parser.add_argument('rosbag_file', action='store', help='Output ROSbag file', type=str)

    parser.add_argument('-p', '--preview', action='store_true')

    args = parser.parse_args()

    simulator = MapSimulator2D(args.map_file, args.robot_file)
    simulator.convert(args.rosbag_file, display=args.preview)
