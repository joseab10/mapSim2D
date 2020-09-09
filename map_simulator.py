#! /usr/bin/env python

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

from collections import deque

# Project Libraries
from geometry import Line, Polygon, rotate2d


def set_dict_param(in_dict, self_dict, key, param_name, default):
    if key in in_dict:
        self_dict[key] = in_dict[key]
    else:
        rospy.logwarn("{} undefined in config file. Using default value: '{}. Help: {}'.".format(key,
                                                                                                 default, param_name))
        self_dict[key] = default


def import_json(in_file, include_path=None, config=None, includes=None, included=None):

    if config is None:
        config = {}
    if include_path is None:
        include_path = []
    if includes is None:
        includes = deque()
    if included is None:
        included = set([])

    import_successful = False

    for path in include_path:
        in_path = path + os.path.sep + in_file
        in_path = os.path.expandvars(in_path)
        in_path = os.path.normpath(in_path)

        if os.path.isfile(in_path):
            try:
                with open(in_path, "r") as f:
                    text = f.read()
                    file_config = json.loads(text)
                    included.add(in_file)
            except (IOError, ValueError):
                rospy.logwarn("Couldn't open %s", in_path)
            else:
                rospy.loginfo("Loaded file %s", in_path)
                import_successful = True
                break

    if not import_successful:
        rospy.logerr("Couldn't open %s in any of the search paths: %s", in_file, ", ".join(include_path))
        exit(-1)

    if "include" in file_config:
        include_list = file_config["include"]
        include_list.reverse()
        includes.extendleft(include_list)
        file_config.pop("include")

    tmp_config = config.copy()

    while includes:
        tmp_file = includes.popleft()
        if tmp_file not in included:
            inc_config = import_json(tmp_file, include_path, config, includes, included)
            tmp_config.update(inc_config)

    tmp_config.update(file_config)

    return tmp_config


class MapSimulator2D:

    def __init__(self, in_file, include_path, override_params=None):

        include_path = include_path.split(os.pathsep)

        config = import_json(in_file, include_path)

        if override_params is not None:
            override_params = json.loads(override_params)
            config.update(override_params)

        # Map Obstacle Parsing
        obstacles = []

        try:
            config_obstacles = config['obstacles']

        except KeyError:
            rospy.logwarn("No obstacles defined in map file")
            config_obstacles = []

        minx = 0
        miny = 0
        maxx = 0
        maxy = 0

        for obstacle in config_obstacles:
            try:
                obstacle_type = obstacle['type']
            except KeyError:
                rospy.logwarn("Obstacle has no type, ignoring")
                continue

            if obstacle_type == "polygon":
                try:
                    vertices = obstacle['vertices']
                    if "opacity" in obstacle:
                        opacity = obstacle['opacity']
                    else:
                        opacity = 1.0
                    new_obstacle = Polygon(vertices, opacity=opacity)
                    obstacles.append(new_obstacle)
                    # Compute display boundaries
                    if new_obstacle.min_x < minx:
                        minx = new_obstacle.min_x
                    if new_obstacle.max_x > maxx:
                        maxx = new_obstacle.max_x
                    if new_obstacle.min_y < miny:
                        miny = new_obstacle.min_y
                    if new_obstacle.max_y > maxy:
                        maxy = new_obstacle.max_y
                except KeyError:
                    rospy.logwarn("Polygon Obstacle has no vertices defined")

            # TODO: Define more types of obstacles
            elif obstacle_type == "circle":
                pass
            elif obstacle_type == "bezier_pline":
                pass

        self._obstacles = obstacles

        self._params = {}

        defaults = {
            "odom_frame":  {"def": 'odom',      "desc": 'Odometry TF Frame'},
            "base_frame":  {"def": "base_link", "desc": "Base TF Frame"},
            "laser_frame": {"def": "base_link", "desc": "Laser TF Frame"},

            "base_to_laser_tf": {"def": [[0, 0], 0], "desc": "Base to Laser Transform"},

            "scan_topic": {"def": "base_scan", "desc": "ROS Topic for scan messages"},

            "deterministic": {"def": False,           "desc": "Deterministic process"},
            "pose_sigma": {"def": [[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]], "desc": "Movement by Pose Covariance Matrix (3x3)"},

            "odometry_alpha": {"def": [0., 0., 0., 0.], "desc": "Movement by Odometry Covariance Matrix (3x3)"},

            "measurement_sigma": {"def": [[0., 0.],
                                          [0., 0.]],     "desc": "Measurement Covariance Matrix (2x2)"},

            "num_rays":  {"def": 180,         "desc": "Number of Sensor beams per scan"},
            "start_ray": {"def": - np.pi / 2, "desc": "Starting scan angle (in Radians)"},
            "end_ray":   {"def":   np.pi / 2, "desc": "End scan angle (in Radians)"},
            "max_range": {"def": 20,          "desc": "Laser Scanner Max. Range (in m)"},

            "meas_per_move": {"def": 5, "desc": "Number of measurements per move command"},

            "initial_timestamp": {"def": None, "desc": "Initial Time Stamp"},

            "move_time_interval": {"def": 1000.0,
                                   "desc": "Time (in ms) that a movement command takes"},
            "scan_time_interval": {"def":   50.0,
                                   "desc": "Time (in ms) that a measurement command takes"},

            "render_move_pause": {"def": 0.5,
                                  "desc": "Time (in s) that the simulation pauses after each move action"},
            "render_sense_pause": {"def": 0.35,
                                   "desc": "Time (in s) that the simulation pauses after each sensing action"},

            "gt_prefix": {"def": "/GT/",
                          "desc": "Prefix for the ground truth pose and measurement topics"}
        }

        # Parse Parameters
        for param, values in defaults.items():
            set_dict_param(config, self._params, param, values['desc'], values['def'])

        # Uncertainty Parameters
        self._params['pose_sigma'] = np.array(self._params['pose_sigma'])
        self._params['odometry_alpha'] = np.array(self._params['odometry_alpha'])
        self._params['measurement_sigma'] = np.array(self._params['measurement_sigma'])

        # Timestamp parameters
        if self._params['initial_timestamp'] is None:
            self._current_time = rospy.Time.from_sec(time())
        else:
            self._current_time = rospy.Time(int(self._params['initial_timestamp']))

        self._moves = []

        # Message Counters
        self._laser_msg_seq = 0
        self._tf_msg_seq = 0

        try:
            self._moves = config['move_commands']
        except KeyError:
            rospy.logwarn("No moves defined in config file. Considering only starting position")

        # Take into account robot moves for display box
        for move in self._moves:
            pos = move["params"][0]
            if pos[0] < minx:
                minx = move[0]
            if pos[0] > maxx:
                maxx = move[0]
            if pos[1] < miny:
                miny = move[1]
            if pos[1] > maxy:
                maxy = move[1]

        # Add a margin of either the max range of the sensor (too large) or just 1m
        # margin = self._params["max_range"] + 1
        margin = 1
        self.min_x = minx - margin
        self.min_y = miny - margin
        self.max_x = maxx + margin
        self.max_y = maxy + margin

        self._noisy_position = np.zeros(2)
        self._noisy_orientation = np.zeros(1)
        self._sensor_position = np.zeros(2)
        self._sensor_orientation = np.zeros(1)
        self._real_position = np.zeros(2)
        self._real_orientation = np.zeros(1)

        try:
            self._noisy_position = np.array(config['start_pose'][0])
            self._real_position = np.copy(self._noisy_position)
        except KeyError:
            rospy.logwarn("No initial position defined in config file. Starting at (0, 0)")

        try:
            self._noisy_orientation = np.array(config['start_pose'][1])
            self._real_orientation = np.copy(self._noisy_orientation)
        except KeyError:
            rospy.logwarn("No initial orientation defined in config file. Starting with theta=0")

        self._compute_sensor_pose()

    def convert(self, output_file, display=False):

        if output_file is None and not display:
            return

        bag = None
        out_path = ""

        if output_file is not None:
            try:
                out_path = os.path.expanduser(output_file)
                bag = rosbag.Bag(out_path, "w")

            except (IOError, ValueError):
                rospy.logerr("Couldn't open %s", output_file)
                exit(-1)

            rospy.loginfo("Writing rosbag to : %s", out_path)

        axes = None

        self._add_tf_msg(bag)
        self._add_tf_msg(bag, ground_truth=True)

        if display:
            plt.ion()
            figure1 = plt.figure(1)
            axes = figure1.add_subplot(111)
            figure1.show()

            self._render(axes, pause=self._params['render_move_pause'])

        for move in self._moves:

            self._move(move)

            self._add_tf_msg(bag)
            self._add_tf_msg(bag, ground_truth=True)

            if display:
                self._render(axes, pause=self._params['render_move_pause'])

            for i in range(int(self._params['meas_per_move'])):
                measurements, endpoints, hits = self._ray_trace()

                if self._params['deterministic']:
                    noisy_meas = measurements
                    meas_noise = np.zeros(2)
                else:
                    meas_noise = np.random.multivariate_normal(np.zeros(2), self._params['measurement_sigma'],
                                                               size=measurements.shape[0])
                    noisy_meas = measurements + meas_noise

                self._add_scan_msg(bag, noisy_meas)
                self._add_scan_msg(bag, measurements, ground_truth=True)

                self._add_tf_msg(bag)
                self._add_tf_msg(bag, ground_truth=True)

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

                    self._render(axes, noisy_endpoints, hits, pause=self._params['render_sense_pause'])
                    self._render(axes, pause=self._params['render_move_pause'] - self._params['render_sense_pause'])

        if bag is not None:
            bag.close()

            rospy.loginfo("Finished simulation and saved to rosbag")

    def _move(self, move_cmd):
        old_pos = np.concatenate((self._noisy_position, self._noisy_orientation))

        if move_cmd['type'] == 'pose':

            if self._params['deterministic']:
                noise = np.zeros_like(old_pos)
            else:
                noise = np.random.multivariate_normal(np.zeros_like(old_pos), self._params['pose_sigma'])

            target_position = np.array(move_cmd['params'][0])
            target_orientation = np.array(move_cmd['params'][1])

            self._real_position = target_position
            self._real_orientation = target_orientation

            self._noisy_position = np.array(target_position + noise[0:1]).flatten()
            self._noisy_orientation = np.array(target_orientation + noise[2]).flatten()

        elif move_cmd['type'] == "odom":

            target_position = np.array(move_cmd['params'][0])
            target_orientation = np.array(move_cmd['params'][1])



            # Compute delta in initial rotation, translation, final rotation
            delta_trans = target_position - self._real_position
            delta_rot1 = np.arctan2(delta_trans[1], delta_trans[0]) - self._real_orientation
            delta_rot2 = target_orientation - self._real_orientation - delta_rot1
            delta_trans = np.matmul(delta_trans, delta_trans)
            delta_trans = np.sqrt(delta_trans)

            delta_rot1_hat = delta_rot1
            delta_trans_hat = delta_trans
            delta_rot2_hat = delta_rot2

            self._real_position = target_position
            self._real_orientation = target_orientation
            # Add Noise
            if not self._params['deterministic']:
                alpha = self._params['odometry_alpha']
                delta_rot1_hat += np.random.normal(0, alpha[0] * np.abs(delta_rot1)
                                                   + alpha[1] * delta_trans)
                delta_trans_hat += np.random.normal(0, alpha[2] * delta_trans
                                                    + alpha[3] * (np.abs(delta_rot1) + np.abs(delta_rot2)))
                delta_rot2_hat += np.random.normal(0, alpha[0] * np.abs(delta_rot2)
                                                   + alpha[1] * delta_trans)

            theta1 = self._noisy_orientation + delta_rot1_hat
            self._noisy_position = self._noisy_position + (delta_trans_hat *
                                                           np.array([np.cos(theta1), np.sin(theta1)]).flatten())
            self._noisy_orientation = theta1 + delta_rot2_hat

        elif move_cmd['type'] == "velocity":
            # TODO
            pass

        # Recompute sensor pose from new robot pose
        self._compute_sensor_pose()
        self._increment_time(self._params['move_time_interval'])

    def _compute_sensor_pose(self):
        tf_trans = np.array(self._params['base_to_laser_tf'][0])
        tf_rot = np.array(self._params['base_to_laser_tf'][1])

        rotation = rotate2d(self._real_orientation)
        translation = rotation.dot(tf_trans)

        self._sensor_position = self._real_position + translation
        self._sensor_orientation = self._real_orientation + tf_rot

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

    def _add_tf_msg(self, bag, ground_truth=False, update_laser_tf=True):

        if bag is None:
            return

        tf2_msg = TFMessage()

        if ground_truth:
            posx = float(self._real_position[0])
            posy = float(self._real_position[1])
            theta = float(self._real_orientation)
            frame_prefix = self._params["gt_prefix"]

            tf_map_odom_msg = TransformStamped()

            tf_map_odom_msg.header.stamp = self._current_time
            tf_map_odom_msg.header.seq = self._tf_msg_seq
            tf_map_odom_msg.header.frame_id = "/map"
            tf_map_odom_msg.child_frame_id = frame_prefix + self._params['odom_frame']

            position = Point(0, 0, 0)
            quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, 0)
            tf_map_odom_msg.transform.translation = position
            tf_map_odom_msg.transform.rotation.x = quaternion[0]
            tf_map_odom_msg.transform.rotation.y = quaternion[1]
            tf_map_odom_msg.transform.rotation.z = quaternion[2]
            tf_map_odom_msg.transform.rotation.w = quaternion[3]

            tf2_msg.transforms.append(tf_map_odom_msg)

        else:
            posx = float(self._noisy_position[0])
            posy = float(self._noisy_position[1])
            theta = float(self._noisy_orientation)
            frame_prefix = ""

        tf_odom_robot_msg = TransformStamped()

        tf_odom_robot_msg.header.stamp = self._current_time
        tf_odom_robot_msg.header.seq = self._tf_msg_seq
        tf_odom_robot_msg.header.frame_id = frame_prefix + self._params['odom_frame']
        tf_odom_robot_msg.child_frame_id = frame_prefix + self._params['base_frame']

        position = Point(posx, posy, 0.0)
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, theta)

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
            tf_laser_robot_msg.header.frame_id = frame_prefix + self._params['base_frame']
            tf_laser_robot_msg.child_frame_id = frame_prefix + self._params['laser_frame']

            position = Point(float(self._params['base_to_laser_tf'][0][0]),
                             float(self._params['base_to_laser_tf'][0][1]), 0.0)
            quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0,
                                                                  float(self._params['base_to_laser_tf'][1][0]))

            tf_laser_robot_msg.transform.translation = position
            tf_laser_robot_msg.transform.rotation.x = quaternion[0]
            tf_laser_robot_msg.transform.rotation.y = quaternion[1]
            tf_laser_robot_msg.transform.rotation.z = quaternion[2]
            tf_laser_robot_msg.transform.rotation.w = quaternion[3]

            tf2_msg.transforms.append(tf_laser_robot_msg)

        bag.write("/tf", tf2_msg, self._current_time)

        if not ground_truth:
            self._tf_msg_seq += 1

    def _add_scan_msg(self, bag, measurements, ground_truth=False):

        if bag is None:
            return

        if ground_truth:
            topic_prefix = self._params["gt_prefix"]
        else:
            topic_prefix = ""

        meas_msg = LaserScan()

        meas_msg.header.frame_id = topic_prefix + self._params['laser_frame']
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

        bag.write(topic_prefix + self._params['scan_topic'], meas_msg, meas_msg.header.stamp)

        if not ground_truth:
            self._laser_msg_seq += 1
            self._increment_time(self._params['scan_time_interval'])

    def _increment_time(self, ms):
        secs = ms
        nsecs = secs
        secs = int(secs / 1000)
        nsecs = int((nsecs - 1000 * secs) * 1e6)

        self._current_time += rospy.Duration(secs, nsecs)

    def _draw_map(self, ax):
        for obstacle in self._obstacles:
            if isinstance(obstacle, Polygon):
                vertices = obstacle.vertices.transpose()
                ax.fill(vertices[0], vertices[1], edgecolor='tab:blue', hatch='////',
                        fill=False, alpha=obstacle.opacity)

    def _draw_robot(self, ax, real=False):
        robot_size = 0.05

        if real:
            pos_x = self._real_position[0]
            pos_y = self._real_position[1]
            theta = self._real_orientation
            orientation_inipt = self._real_position
            robot_color = "tab:green"
            z_order = 4
        else:
            pos_x = self._noisy_position[0]
            pos_y = self._noisy_position[1]
            theta = self._noisy_orientation
            orientation_inipt = self._noisy_position
            robot_color = "tab:purple"
            z_order = 2

        robot_base = plt.Circle((pos_x, pos_y), robot_size, color=robot_color, zorder=z_order)
        ax.add_artist(robot_base)

        orientation_endpt = robot_size * np.array([np.cos(theta), np.sin(theta)]).reshape((2,))
        orientation_endpt += orientation_inipt
        orientation_line = np.stack((orientation_inipt, orientation_endpt)).transpose()

        robot_orientation = plt.Line2D(orientation_line[0], orientation_line[1], color='white', zorder=z_order + 1)
        ax.add_artist(robot_orientation)

        return robot_base

    def _draw_beams(self, ax, beams, hits):
        if beams is None:
            return

        for i, beam in enumerate(beams):
            ray = np.array([self._sensor_position, beam])
            ray = ray.transpose()

            if hits[i]:
                ax.plot(ray[0], ray[1], 'tab:red', marker='.', linewidth=0.7)
            else:
                ax.plot(ray[0], ray[1], 'tab:red', marker='1', dashes=[10, 6], linewidth=0.5)

    def _render(self, ax, beam_endpoints=None, hits=None, pause=0.25):
        """
        Renders a graphical view of the map, the current state of the robot and the measurements using Matplotlib

        :param ax: Axes object so as to not redefine and recreate it every time a frame is rendered
        :param beam_endpoints: Numpy array of the laser endpoints. None if measurements are not to be displayed.
        :param hits: Numpy array of booleans indicating whether each laser beam actually hit an obstacle or was a
        max. range reading.

        :return: None
        """

        ax.clear()

        ax.set_aspect('equal', 'box')
        ax.set_xlim([self.min_x, self.max_x])
        ax.set_xbound(self.min_x, self.max_x)
        ax.set_ylim(self.min_y, self.max_y)
        ax.set_ybound(self.min_y, self.max_y)

        self._draw_map(ax)
        self._draw_beams(ax, beam_endpoints, hits)
        noisy_robot_handle = self._draw_robot(ax)
        real_robot_handle = self._draw_robot(ax, real=True)

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#CCCCCC')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')

        ax.legend((real_robot_handle, noisy_robot_handle), ("Real Pose", "Noisy Odometry"),
                  loc='lower center')

        ax.grid(True)

        plt.draw()
        plt.pause(0.0001)

        sleep(pause)


if __name__ == '__main__':
    import argparse

    rospy.init_node('map_simulator', anonymous=True)

    parser = argparse.ArgumentParser(description="Generate a ROSbag file from a simulated robot trajectory.")

    parser.add_argument('-i', '--input', action='store', help='Input JSON robot config file', type=str)
    parser.add_argument('-o', '--output', action='store', help='Output ROSbag file', type=str, required=False)

    parser.add_argument('-p', '--preview', action='store_true')
    parser.add_argument('-s', '--search_paths', action='store', type=str, default='.:robots:maps',
                        help='Search paths for the input and include files separated by colons (:)')

    args, override_args = parser.parse_known_args()

    override_str = None

    if len(override_args) > 0:
        override_str = '{'
        for arg in override_args:
            arg_keyval = arg.split(":=")
            override_str += '"' + str(arg_keyval[0]) + '":' + str(arg_keyval[1]) + ','

        override_str = override_str[0:-1] + "}"

    simulator = MapSimulator2D(args.input, args.search_paths, override_params=override_str)
    simulator.convert(args.output, display=args.preview)
