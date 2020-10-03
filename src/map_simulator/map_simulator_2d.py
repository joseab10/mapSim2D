# ROS Libraries
import rospy
from tf.transformations import quaternion_from_euler
import rosbag

# ROS Messages
from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

import os.path
from time import sleep, time
import sys

import simplejson as json

# Math Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from collections import deque

# Project Libraries
from map_simulator.geometry.primitives import Line, Polygon
from map_simulator.geometry.transform import rotate2d
from map_simulator.utils import tf_frame_normalize, tf_frame_join


class MapSimulator2D:
    """
    Class for simulating a robot's pose and it's scans using a laser sensor given a series of move commands around a map
    defined with polygonal obstacles.
    """

    def __init__(self, in_file, include_path, override_params=None):
        """
        Constructor

        :param in_file: (string) Name of the main robot parameter file. Actual file might include more files.
        :param include_path: (list) List of path strings to search for main and included files.
        :param override_params: (dict) Dictionary of parameter:value pairs to override any configuration
                                       defined in the files.
        """

        rospy.init_node('mapsim2d', anonymous=True)

        config = self._import_json(in_file, include_path)

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
            "map_frame":   {"def": "map",       "desc": "Map TF Frame"},
            "odom_frame":  {"def": "odom",      "desc": "Odometry TF Frame"},
            "base_frame":  {"def": "base_link", "desc": "Base TF Frame"},
            "laser_frame": {"def": "laser_link", "desc": "Laser TF Frame"},

            "gt_prefix":  {"def": "GT",  "desc": "Ground Truth TF prefix for the pose and measurement topics"},
            "odo_prefix": {"def": "odo", "desc": "Odometry TF prefix for the pose and measurement topics"},

            "base_to_laser_tf": {"def": [[0, 0], 0], "desc": "Base to Laser Transform"},

            "scan_topic": {"def": "base_scan", "desc": "ROS Topic for scan messages"},

            "deterministic": {"def": False,           "desc": "Deterministic process"},

            "move_noise_type": {"def": "odom", "desc": "Type of movement noise [linear|odom]"},
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
                                   "desc": "Time (in s) that the simulation pauses after each sensing action"}
        }

        # Parse Parameters
        for param, values in defaults.items():
            self._set_dict_param(config, self._params, param, values['desc'], values['def'])

        # Uncertainty Parameters
        self._params['pose_sigma'] = np.array(self._params['pose_sigma'])
        self._params['odometry_alpha'] = np.array(self._params['odometry_alpha'])
        self._params['measurement_sigma'] = np.array(self._params['measurement_sigma'])

        # Timestamp parameters
        if self._params['initial_timestamp'] is None:
            self._current_time = rospy.Time.from_sec(time())
        else:
            self._current_time = rospy.Time(int(self._params['initial_timestamp']))

        # Message Counters
        self._laser_msg_seq = 0
        self._tf_msg_seq = 0

        # Initial Position and Orientation
        self._noisy_position = np.zeros(2)
        self._noisy_orientation = np.zeros(1)
        self._real_position = np.zeros(2)
        self._real_orientation = np.zeros(1)
        self._real_sensor_position = np.zeros(2)
        self._real_sensor_orientation = np.zeros(1)

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

        moves = []
        try:
            moves = config['move_commands']
        except KeyError:
            rospy.logwarn("No moves defined in config file. Considering only starting position")

        # Generate Pose list from move commands
        pp = self._real_position
        po = self._real_orientation
        self._pose_list = [[pp, po]]
        for move in moves:
            poses = self._get_poses(move, prev_pose=pp, prev_orientation=po)
            if poses is not None:
                for pose in poses:
                    pp = pose[0]
                    po = pose[1]
                    self._pose_list.append([pp, po])

                    # Take into account robot moves for display box
                    if pp[0] < minx:
                        minx = move[0]
                    if pp[0] > maxx:
                        maxx = move[0]
                    if pp[1] < miny:
                        miny = move[1]
                    if pp[1] > maxy:
                        maxy = move[1]

        # Add a margin of either the max range of the sensor (too large) or just 1m
        # margin = self._params["max_range"] + 1
        margin = 1
        self._min_x = minx - margin
        self._min_y = miny - margin
        self._max_x = maxx + margin
        self._max_y = maxy + margin

    @staticmethod
    def _set_dict_param(in_dict, self_dict, key, param_name, default):
        """
        Method for setting a parameter's value, or take the default one if not provided.

        :param in_dict: (dict) Input dictionary from which (valid) parameters will be read.
        :param self_dict: (dic) Own dictionary to which only valid parametres will be written.
        :param key: (string) Parameter key (name) under which the value is stored both in the in_dict and self_dict.
        :param param_name: (string) Parameter long name to be displayed in output messages.
        :param default: Default value to be used in case parameter is not defined in in_dict.

        :return: None
        """

        if key in in_dict:
            self_dict[key] = in_dict[key]
        else:
            rospy.logwarn("{} undefined in config file. Using default value: '{}. Help: {}'.".format(key,
                                                                                                     default,
                                                                                                     param_name))
            self_dict[key] = default

    def _import_json(self, in_file, include_path=None, config=None, includes=None, included=None):
        """
        Recursively import a JSON file and it's included subfiles.

        :param in_file: (string) Path string pointing to a JSON configuration file.
        :param include_path: (list) List of paths where to search for files.
        :param config: (dict) Configuration dictionary to add parameters to.
        :param includes: (deque) Queue of files to be included as read from the file's include statement.
        :param included: (set) Set of files already included to prevent multiple inclusions.

        :return: (dict) Dictionary of settings read from the JSON file(s).
        """

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
            in_path = os.path.join(path, in_file)
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
                inc_config = self._import_json(tmp_file, include_path, config, includes, included)
                tmp_config.update(inc_config)

        tmp_config.update(file_config)

        return tmp_config

    def simulate(self, output_file, display=False):
        """
        Main method for simulating and either saving the data to a ROSBag file, displaying it or both.

        :param output_file: (string|None) Path and filename to where the ROSBag file is to be saved.
                                          If None, then nothing will be saved to file, only simulated.
        :param display: (bool) Display the simulation. Set to False to run faster
                               (no pauses in the execution for visualization)

        :return: None
        """

        if output_file is None and not display:
            return

        bag = None
        out_path = ""

        if output_file is not None:
            try:
                out_path = os.path.expandvars(os.path.expanduser(output_file))
                bag = rosbag.Bag(out_path, "w")

            except (IOError, ValueError):
                rospy.logerr("Couldn't open %s", output_file)
                exit(-1)

            rospy.loginfo("Writing rosbag to : %s", out_path)

        axes = None

        if display:
            plt.ion()
            figure1 = plt.figure(1)
            axes = figure1.add_subplot(111)
            figure1.show()

        meas_num = int(self._params['meas_per_move'])
        pose_len = float(len(self._pose_list))

        self._add_tf_messages(bag)

        for p_cnt, pose in enumerate(self._pose_list):

            # For each scan while stopped
            for _ in range(meas_num):
                meas, noisy_meas = self._scan(display=display, axes=axes)

                self._add_scan_messages(bag, noisy_meas, det_measurements=meas)
                # Publish pose after each measurement, otherwise gmapping doesn't process the scans
                self._add_tf_messages(bag)

            # Move robot to new pose(s)
            self._move(pose, display=display, axes=axes)
            self._print_status(float(p_cnt) / pose_len)

        sys.stdout.write('\n')
        sys.stdout.flush()

        rospy.loginfo("Finished simulation")
        if bag is not None:
            bag.close()
            rospy.loginfo("Rosbag saved and closed")

    def _get_poses(self, move_cmd, prev_pose=None, prev_orientation=None):
        """
        Generates the pose(s) [[x, y], theta] from move command dictionaries

        :param move_cmd: (dict) Movement command dictionary with format:
                                {"type": <mv_type>, "<par1_key>": <par1_val>, "<par2_key>": <par2_val>, ...}
                                Supported types are:
                                    * "comment": Ignored, used just for commenting JSON files
                                    * "pose": Directly returns the pose given by "params": [[x, y], th]
                                    * "odom": Directly returns the pose given by "params": [[x, y], th]
                                    * "linear": Linear interpolation between "start": [x0, y0] (optional)
                                                and "end": [x1, y1] with "steps": n steps.
                                                If "start" is not given, current position is used as start point.
                                                Theta is defined as the angle of the line w.r.t. x for the entire move.
                                    * "interpolation": Linear interpolation between starting pose "start": [x0, y0]
                                                       and "start_angle": th0, and end pose "end": [x1, y1]
                                                       and "end_angle": th1, in "steps": n steps.
                                                       Unlike "linear", the angle is also linearly interpolated between
                                                       start and end poses.
                                                       If "start" and "start_angle" are omitted, current pose is used.
                                    * "rotation": In-place rotation without movement between "start": th0 (optional)
                                                  and "end": th1 in direction "dir": "cw"|"ccw" (clockwise or
                                                  counterclockwise) in "steps": n steps.
                                                  If "start" is omitted, current orientation is used as starting point.
                                                  Current pose is preserved during each step.
                                    * "circular": Circular path from "start": [x0, y0] (optional)
                                                  ending in "end": [x1, y1] around "center": [cx0, cy0]
                                                  in direction "dir": "cw"|"ccw" (clockwise or counterclockwise)
                                                  in "steps": n steps.
                                                  Because of possible errors between center, start and end, the turning
                                                  radius is the average between the center-start and center-end lengths.
                                                  If "start" is omitted, current pose is used.
                                                  Angle th is computed as tangential to the circular path in each step.

        :return: (list) List of poses [..., [[x, y], th], ...]
        """

        mtype = str(move_cmd['type']).lower()

        if prev_pose is None:
            prev_pose = self._real_position
        if prev_orientation is None:
            prev_orientation = self._real_orientation

        prev_pose = np.array(prev_pose)

        if mtype == "pose" or mtype == "odom":
            return move_cmd['params']

        if mtype == "linear":
            if "start" in move_cmd:
                start = np.array(move_cmd['start'])
                rm_first_row = False
            else:
                start = prev_pose
                rm_first_row = True

            end = np.array(move_cmd['end'])
            steps = move_cmd['steps']

            diff = end - start
            theta = np.arctan2(diff[1], diff[0])
            if theta > np.pi:
                theta -= 2 * np.pi
            if theta < -np.pi:
                theta += 2 * np.pi

            poses = np.linspace(start, end, num=steps)
            poses = [[[p[0], p[1]], theta] for p in poses]

            if rm_first_row:
                poses = poses[1:]

            return poses

        if mtype == "rotation":
            if "start" in move_cmd:
                start = move_cmd['start']
                rm_first_row = False
            else:
                start = prev_orientation
                rm_first_row = True

            cw = True
            if "dir" in move_cmd:
                if move_cmd['dir'] == "ccw":
                    cw = False

            end = move_cmd['end']
            steps = move_cmd['steps']

            if cw and start < end:
                end -= 2 * np.pi
            if not cw and start > end:
                end += 2 * np.pi

            theta = np.linspace(start, end, num=steps)
            theta[theta > np.pi] -= 2 * np.pi
            theta[theta < -np.pi] += 2 * np.pi

            poses = [[prev_pose, th] for th in theta]

            if rm_first_row:
                poses = poses[1:]

            return poses

        if mtype == "circular":
            if "start" in move_cmd:
                start = np.array(move_cmd['start'])
                rm_first_row = False
            else:
                start = prev_pose
                rm_first_row = True

            cw = True
            if "dir" in move_cmd:
                if move_cmd['dir'] == "ccw":
                    cw = False

            end = np.array(move_cmd['end'])
            center = np.array(move_cmd['center'])
            steps = move_cmd['steps']

            start_diff = start - center
            start_angle = np.arctan2(start_diff[1], start_diff[0])
            end_diff = end - center
            end_angle = np.arctan2(end_diff[1], end_diff[0])

            if cw and start_angle < end_angle:
                end_angle -= 2 * np.pi
            if not cw and start_angle > end_angle:
                end_angle += 2 * np.pi

            radius = np.sqrt(np.dot(start_diff, start_diff))
            radius += np.sqrt(np.dot(end_diff, end_diff))
            radius /= 2

            angles = np.linspace(start_angle, end_angle, num=steps)

            poses_x = center[0] + radius * np.cos(angles)
            poses_y = center[1] + radius * np.sin(angles)
            thetas = angles + np.pi / 2
            thetas[thetas > np.pi] -= 2 * np.pi
            thetas[thetas < -np.pi] += 2 * np.pi

            poses = [[[poses_x[i], poses_y[i]], thetas[i]] for i in range(angles.shape[0])]

            if rm_first_row:
                poses = poses[1:]

            return poses

        if mtype == "interpolation":
            start = prev_pose
            start_theta = prev_orientation
            end = move_cmd['end']
            end_theta = move_cmd['end_angle']
            steps = move_cmd['steps']

            poses = np.linspace(start, end, num=steps)
            thetas = np.linspace(start_theta, end_theta, num=steps)

            poses = [[poses[i], thetas[i]] for i in range(thetas.shape[0])]
            poses = poses[1:]

            return poses

        return None

    def _move(self, pose, display=False, axes=None):
        """
        Take a real pose and add noise to it. Then recompute the laser sensor pose and increment the time.
        If the parameter "deterministic" is set to True, then both the real and noisy poses will be equal.
        Otherwise, tne noisy pose will be computed depending on the "move_noise_type" parameter, which can be:
            * "odom": The movement is modeled as an initial rotation, then a translation and finally another rotation.
                      Noise is added to each of these steps as a weighted sum of the actual displacements with weights
                      defined by "odometry_alpha": [a1, a2, a3, a4].
            * "linear": Zero-mean gaussian noise with covariance matrix
                        "pose_sigma": [[sxx, sxy, sxth], [syx, syy, syth], [sthx, sthy, sthth]]
                        is added to the target pose.

        :param pose: (list) Next pose of the robot defined as [[x, y], th]
        :param display: (bool)[Default: False] Display pose and beams using matplotlib if True.
        :param axes: [Default: None] Matplotlib axes object to draw to.

        :return: None
        """

        target_position = np.array(pose[0])
        target_orientation = np.array(pose[1])

        if self._params['deterministic'] or self._tf_msg_seq == 0:
            self._noisy_position = target_position
            self._noisy_orientation = target_orientation

        else:
            # Rotation/Translation/Rotation error
            if self._params['move_noise_type'] == "odom":

                # Compute delta in initial rotation, translation, final rotation
                delta_trans = target_position - self._real_position
                delta_rot1 = np.arctan2(delta_trans[1], delta_trans[0]) - self._real_orientation
                delta_rot2 = target_orientation - self._real_orientation - delta_rot1
                delta_trans = np.matmul(delta_trans, delta_trans)
                delta_trans = np.sqrt(delta_trans)

                delta_rot1_hat = delta_rot1
                delta_trans_hat = delta_trans
                delta_rot2_hat = delta_rot2

                # Add Noise
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

            # Linear error
            else:

                noise = np.random.multivariate_normal(np.zeros(3), self._params['pose_sigma'])

                self._noisy_position = np.array(target_position + noise[0:1]).flatten()
                self._noisy_orientation = np.array(target_orientation + noise[2]).flatten()

        self._real_position = target_position
        self._real_orientation = target_orientation

        # Recompute sensor pose from new robot pose
        self._compute_sensor_pose()
        self._increment_time(self._params['move_time_interval'])
        self._tf_msg_seq += 1

        if display and (axes is not None):
            move_pause = float(self._params['render_move_pause'])
            self._render(axes, pause=move_pause)

    def _scan(self, display=False, axes=None):
        """
        Generate a scan using ray tracing, add noise if configured and display it.

        :param display: (bool)[Default: False] Display pose and beams using matplotlib if True.
        :param axes: [Default: None] Matplotlib axes object to draw to.

        :return: (np.ndarray, np.ndarray) Arrays of Measurements and Noisy Measurements respectively to be
                                          displayed and published.
                                          Each 2D array is comprised of [bearings, ranges].
        """

        meas, endpoints, hits = self._ray_trace()

        if self._params['deterministic']:
            noisy_meas = meas
            meas_noise = np.zeros(2)
        else:
            meas_noise = np.random.multivariate_normal(np.zeros(2), self._params['measurement_sigma'],
                                                       size=meas.shape[0])
            noisy_meas = meas + meas_noise

        if display and (axes is not None):
            if self._params['deterministic']:
                noisy_endpoints = endpoints
            else:
                range_noises = meas_noise[:, 1]
                bearing_noises = meas_noise[:, 0]
                bearing_noises = np.array([np.cos(bearing_noises), np.sin(bearing_noises)])
                endpoint_noise = range_noises * bearing_noises
                endpoint_noise = endpoint_noise.transpose()
                noisy_endpoints = endpoints + endpoint_noise

            meas_pause = float(self._params['render_sense_pause'])
            meas_off_pause = float(self._params['render_move_pause'] - meas_pause)
            self._render(axes, noisy_endpoints, hits, pause=meas_pause)
            self._render(axes, pause=meas_off_pause)

        # Increment time and sequence
        self._increment_time(self._params['scan_time_interval'])
        self._laser_msg_seq += 1

        return meas, noisy_meas

    def _compute_sensor_pose(self):
        """
        Computes the real sensor pose from the real robot pose and the base_to_laser_tf transform.

        :return: None
        """
        tf_trans = np.array(self._params['base_to_laser_tf'][0])
        tf_rot = np.array(self._params['base_to_laser_tf'][1])

        rotation = rotate2d(self._real_orientation)
        translation = rotation.dot(tf_trans)

        self._real_sensor_position = self._real_position + translation
        self._real_sensor_orientation = self._real_orientation + tf_rot

    def _ray_trace(self):
        """
        Generates a set of laser measurements starting from the laser sensor pose until the beams either hit an obstacle
        or reach the maximum range.

        :return: (tuple) Tuple comprised of:
                             * bearing_ranges: ndarray of measurement angles
                             * endpoints: ndarray of (x,y) points where the laser hit or reached max_range,
                             * hits: ndarray of boolean values stating whether the beam hit an obstacle.
                                     True for hit, False for max_range measurement.
        """

        bearing_ranges = []
        endpoints = []
        hits = []

        bearing = self._params['start_ray']

        num_rays = self._params['num_rays']
        if num_rays > 1:
            num_rays -= 1

        bearing_increment = (self._params['end_ray'] - self._params['start_ray']) / num_rays

        for i in range(int(self._params['num_rays'])):

            theta = self._real_sensor_orientation + bearing
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([c, s]).flatten()
            max_ray_endpt = self._real_sensor_position + self._params['max_range'] * rotation_matrix
            ray = Line(self._real_sensor_position, max_ray_endpt)

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

    @staticmethod
    def __add_transform(msg, ts, seq, p_frame, c_frame, position, rotation):
        """
        Function for appending a transform to an existing TFMessage.

        :param msg: (tf2_msgs.TFMessage) A TF message to append a transform to.
        :param ts: (rospy.Time) A time stamp for the Transform's header.
        :param seq: (int) The number/sequence of the TF message.
        :param p_frame: (string) Parent frame of the transform.
        :param c_frame: (string) Child frame of the transform.
        :param position: (geometry_msg.Point) A point representing the translation between frames.
        :param rotation: (list) A quaternion representing the rotation between frames.

        :return: None
        """

        tran = TransformStamped()

        tran.header.stamp = ts
        tran.header.seq = seq
        tran.header.frame_id = p_frame
        tran.child_frame_id = c_frame

        tran.transform.translation = position
        tran.transform.rotation.x = rotation[0]
        tran.transform.rotation.y = rotation[1]
        tran.transform.rotation.z = rotation[2]
        tran.transform.rotation.w = rotation[3]

        msg.transforms.append(tran)

    def __add_tf_msg(self, msg, real_pose=False, tf_prefix="", update_laser_tf=True,
                    publish_map_odom=False):
        """
        Appends the tf transforms to the passed message with the real/noisy pose of the robot
        and (optionally) the laser sensor pose.

        :param msg: (tf2_msgs.TFMessage) A TF message to append all the transforms to.
        :param real_pose: (bool)[Default: False] Publish real pose if True, noisy pose if False.
        :param tf_prefix: (string)[Default: ""] Prefix to be prepended to each TF Frame
        :param update_laser_tf: (bool)[Default: True] Publish base_link->laser_link tf if True.
        :param publish_map_odom: (bool)[Default: False] Publish the map->odom tf transform if True.

        :return: None
        """

        ts = self._current_time
        seq = self._tf_msg_seq

        odom_frame = tf_frame_normalize(tf_frame_join(tf_prefix, str(self._params['odom_frame'])))
        base_frame = tf_frame_normalize(tf_frame_join(tf_prefix, str(self._params['base_frame'])))

        if publish_map_odom:
            map_frame = str(self._params['map_frame'])
            zero_pos = Point(0.0, 0.0, 0.0)
            zero_rot = quaternion_from_euler(0.0, 0.0, 0.0)

            self.__add_transform(msg, ts, seq, map_frame, odom_frame, zero_pos, zero_rot)

        if real_pose:
            pos_x = float(self._real_position[0])
            pos_y = float(self._real_position[1])
            theta = float(self._real_orientation)

        else:
            pos_x = float(self._noisy_position[0])
            pos_y = float(self._noisy_position[1])
            theta = float(self._noisy_orientation)

        odom_pos = Point(pos_x, pos_y, 0.0)
        odom_rot = quaternion_from_euler(0.0, 0.0, theta)

        self.__add_transform(msg, ts, seq, odom_frame, base_frame, odom_pos, odom_rot)

        if update_laser_tf:
            laser_frame = tf_frame_normalize(tf_frame_join(tf_prefix, str(self._params['laser_frame'])))

            lp_x = float(self._params['base_to_laser_tf'][0][0])
            lp_y = float(self._params['base_to_laser_tf'][0][1])
            lp_th = float(self._params['base_to_laser_tf'][1][0])

            laser_pos = Point(lp_x, lp_y, 0.0)
            laser_rot = quaternion_from_euler(0.0, 0.0, lp_th)

            self.__add_transform(msg, ts, seq, base_frame, laser_frame, laser_pos, laser_rot)

    def _add_tf_messages(self, bag, add_gt=True, add_odom=True, update_laser_tf=True):
        """
        Function for adding all TF messages (Noisy pose, GT pose, Odom pose) at once.

        :param bag: (rosbag.Bag) Open ROSBag file handler where the messages will be stored.
        :param add_gt: (bool)[Default: True] Publish the ground truth transforms.
        :param add_odom (bool)[Default: True] Publish the plain odometry transforms.
        :param update_laser_tf: (bool)[Default: True] Publish base_link->laser_link tf if True.

        :return: None
        """

        if bag is None:
            return

        tf2_msg = TFMessage()

        # Noisy Pose
        first_msg = self._tf_msg_seq == 0
        self.__add_tf_msg(tf2_msg, real_pose=False, tf_prefix="", update_laser_tf=update_laser_tf,
                          publish_map_odom=first_msg)

        # Ground Truth Pose
        if add_gt:
            gt_prefix = str(self._params['gt_prefix'])
            self.__add_tf_msg(tf2_msg, real_pose=True, tf_prefix=gt_prefix, update_laser_tf=update_laser_tf,
                              publish_map_odom=True)

        # Odometry Pose
        if add_odom:
            odo_prefix = str(self._params['odo_prefix'])
            self.__add_tf_msg(tf2_msg, real_pose=False, tf_prefix=odo_prefix, update_laser_tf=update_laser_tf,
                              publish_map_odom=True)

        bag.write("/tf", tf2_msg, self._current_time)
        bag.flush()

    def __add_scan_msg(self, bag, topic, frame, measurements):
        """
        Publish a LaserScan message to a ROSBag file with the given measurement ranges.

        :param bag: (rosbag.Bag) Open ROSBag file handler where the message will be stored.
        :param measurements: (ndarray) 2D Array of measurements to be published.
                                       Measured ranges must be in measurements[:, 1]

        :return: None
        """

        if bag is None:
            return

        meas_msg = LaserScan()

        meas_msg.header.frame_id = frame  # topic_prefix + self._params['laser_frame']
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
        meas_msg.intensities = 4096 * (measurements[:, 1] / self._params['max_range'])

        bag.write(topic, meas_msg, self._current_time)
        bag.flush()

    def _add_scan_messages(self, bag, noisy_measurements, det_measurements=None, add_gt=True, add_odom=True):
        """
        Function for adding all scan messages (Noisy pose, GT pose, Odom pose) at once.

        :param bag: (rosbag.Bag) Open ROSBag file handler where the messages will be stored.
        :param noisy_measurements: (ndarray) 2D Array of noisy measurements to be published.
                                             Measured ranges must be in measurements[:, 1]
       :param det_measurements: (ndarray)[Default: None] 2D Array of deterministic measurements to be published.
                                                         Measured ranges must be in measurements[:, 1].
                                                         If None, then ground truth won't be published,
                                                         even if add_gt is True.
       :param add_gt: (bool)[Default: True] Add scan message from ground truth pose frame if True
                                            and det_measurements is not None.
       :param add_odom: (bool)[Default: True] Add scan messages from odom pose frame if True.

       :return: None
        """

        if bag is None:
            return

        topic = str(self._params['scan_topic'])
        frame = str(self._params['laser_frame'])

        self.__add_scan_msg(bag, topic, frame, noisy_measurements)

        if add_gt and det_measurements is not None:
            gt_prefix = str(self._params['gt_prefix'])
            gt_topic = '/' + tf_frame_normalize(tf_frame_join(gt_prefix, topic))
            gt_frame = tf_frame_normalize(tf_frame_join(gt_prefix, frame))

            self.__add_scan_msg(bag, gt_topic, gt_frame, det_measurements)

        if add_odom:
            odo_prefix = str(self._params['odo_prefix'])
            odo_topic = '/' + tf_frame_normalize(tf_frame_join(odo_prefix, topic))
            odo_frame = tf_frame_normalize(tf_frame_join(odo_prefix, frame))

            self.__add_scan_msg(bag, odo_topic, odo_frame, noisy_measurements)

    def _increment_time(self, ms):
        """
        Increment the internal simulated time variable by a given amount.

        :param ms: (float) Time in miliseconds to increment the time.

        :return: None
        """

        secs = ms
        nsecs = secs
        secs = int(secs / 1000)
        nsecs = int((nsecs - 1000 * secs) * 1e6)

        self._current_time += rospy.Duration(secs, nsecs)

    def _draw_map(self, ax):
        """
        Draw the map's obstacles.

        :param ax: Matplotlib axes object to draw to.

        :return: None
        """

        for obstacle in self._obstacles:
            if isinstance(obstacle, Polygon):
                vertices = obstacle.vertices.transpose()
                ax.fill(vertices[0], vertices[1], edgecolor='tab:blue', hatch='////',
                        fill=False, alpha=obstacle.opacity)

    def _draw_robot(self, ax, real=False):
        """
        Draw the robot's real/noisy pose.

        :param ax: Matplotlib axes object to draw to.
        :param real: (bool) Draw the real pose if True, the noisy one if False.

        :return: None
        """
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
        """
        Draw each of the laser measurement beams from the robot's real pose.

        :param ax: Matplotlib axes object to draw to.
        :param beams: (ndarray|None) List of the beams' endpoints (x, y). None to not display the measurements.
        :param hits: (ndarray) List of booleans stating for each beam whether it hit an obstacle (True)
                               or reached max_range (False).

        :return: None
        """
        if beams is None:
            return

        for i, beam in enumerate(beams):
            ray = np.array([self._real_sensor_position, beam])
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
        ax.set_xlim([self._min_x, self._max_x])
        ax.set_xbound(self._min_x, self._max_x)
        ax.set_ylim(self._min_y, self._max_y)
        ax.set_ybound(self._min_y, self._max_y)

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

    @staticmethod
    def _print_status(percent, length=40):
        """
        Prints the percentage status of the simulation.

        :param percent: (float) The percentage to be displayed numerically and in the progress bar.
        :param length: (int)[Default: 40] Length in characters that the progress bar will measure.

        :return: None
        """

        # Erase line and move to the beginning
        sys.stdout.write('\x1B[2K')
        sys.stdout.write('\x1B[0E')

        progress = "Simulation Progress: ["

        for i in range(0, length):
            if i < length * percent:
                progress += '#'
            else:
                progress += ' '
        progress += "] " + str(round(percent * 100.0, 2)) + "%"

        sys.stdout.write(progress)
        sys.stdout.flush()
