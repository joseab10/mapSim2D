# ROS Libraries
import rospy
import tf
import tf2_ros
from tf import TransformerROS
from tf.transformations import quaternion_multiply, quaternion_conjugate, decompose_matrix, quaternion_from_euler

# ROS Message Libraries
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64

# Math Libraries
import numpy as np

# OS Libraries
import os
import os.path
import datetime


def frame_eq(tf1, tf2):
    """
    Function for determining whether two TF chains are equal by ignoring slashes

    :param tf1: (string) First TF frame chain
    :param tf2: (string) Second TF frame chain

    :return: (bool) True if tf1 and tf2 represent the same path ignoring slashes
    """

    tf1_list = filter(None, tf1.split('/'))
    tf2_list = filter(None, tf2.split('/'))

    eq = tf1_list == tf2_list
    return eq


def quaternion_axis_angle(q):
    """
    Convert a rotation expressed as a quaternion into a 3D vector
    representing the axis of rotation and the rotation angle in radians.

    :param q: (list|tuple) A quaternion (x, y, z, w)

    :return: (numpy.array, float) A tuple containting a 3D numpy vector and the angle as float
    """

    w, v = q[3], q[0:2]
    theta = np.arccos(w) * 2
    return v, theta


class PoseErrorCalculator:

    def __init__(self):
        """
        Initialize the PoseErrorCalculator object, the ROS node,
        get the parameters and keep the node alive (spin)
        """

        rospy.init_node('pose_error_calc')

        # Error Accumulators
        self._cum_trans_err = 0
        self._cum_rot_err = 0
        self._cum_tot_err = 0

        # Buffers for last step's poses and info
        ini_pose = ([0.0, 0.0, 0.0], [0.0, 0.0, 0., 0.0])
        self._last_gt_pose = ini_pose
        self._last_sl_mo_pose = ini_pose
        self._last_sl_ob_pose = ini_pose

        self._last_pose_acq = False
        self._last_seq = -1
        self._last_ts = None

        # Weight factor for rotational error
        self._lambda = rospy.get_param("~lambda", 0.1)

        # Publish and save boolean options
        self._publish_error = rospy.get_param("~pub_err", True)
        self._log_error = rospy.get_param("log_err", True)

        if not (self._publish_error or self._log_error):
            rospy.logerr("Not publising nor logging. Why call me then? Exiting.")
            return

        # TF Frames
        self._map_frame = rospy.get_param("~map_frame", "map")
        self._odom_frame = rospy.get_param("~odom_frame", "odom")
        self._base_frame = rospy.get_param("~base_frame", "base_link")
        self._gt_odom_frame = rospy.get_param("~gt_odom_frame", "GT/odom")
        self._gt_base_frame = rospy.get_param("~gt_base_frame", "GT/base_link")

        # CSV Log Path parameters
        default_path = os.path.join("~", "Desktop")
        default_path = os.path.join(default_path, "FMP_logs")
        log_dir = rospy.get_param("log_dir", default_path)
        log_dir = os.path.expandvars(os.path.expanduser(log_dir))
        # CSV Log File parameters
        log_prefix = rospy.get_param("~log_prefix", "pose_err")
        timestamp = datetime.datetime.now()
        timestamp = datetime.datetime.strftime(timestamp, "%y%m%d_%H%M%S")
        log_file = log_prefix + timestamp + '.csv'
        self._log_file = os.path.join(log_dir, log_file)
        # CSV row and column delimiters
        self._newline = rospy.get_param("~newline", "\n")
        self._delim = rospy.get_param("~delim", ",")

        if self._log_error:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            self._init_log()

        # Subscribers / Publishers
        rospy.Subscriber("/tf", TFMessage, self._tf_callback)
        self._tf_listener = tf.TransformListener()

        if self._publish_error:
            self._err_publisher = rospy.Publisher("/pose_err", Float64, latch=True, queue_size=1)

        rospy.spin()

    def _tf_callback(self, msg):
        """
        Function called whenever a TF message is received.
        It tries to store the transforms for the Ground Truth pose (map->GT/base_link),
        and the SLAM pose (map->odom, odom->base_link), and only process them once the
        ground truth message's sequence changes to use the latest poses and allow the
        the SLAM algorithm to register all scans and compute the correction transform.
        If configured, it will publish the error as a float64 message and log to a CSV file.

        :param msg: (tf2_msgs.TFMessage) TF Messages

        :return: None
        """

        pose_acq = False
        seq_chgd = False

        gt_pose = None
        sl_mo_pose = None
        sl_ob_pose = None

        seq = -1
        time_stamp = None

        lookup_ts = rospy.Time(0)  # Latest
        # lookup_ts = rospy.Time.now()
        timeout = rospy.Duration.from_sec(1)

        for transform in msg.transforms:
            seq = transform.header.seq
            time_stamp = transform.header.stamp
            pframe = transform.header.frame_id
            cframe = transform.child_frame_id

            is_gt_pose = frame_eq(pframe, self._map_frame) and frame_eq(cframe, self._gt_odom_frame)

            if is_gt_pose:
                seq_chgd = self._last_seq != seq

                try:
                    self._tf_listener.waitForTransform(self._map_frame, self._gt_base_frame, lookup_ts, timeout)
                    gt_pose = self._tf_listener.lookupTransform(self._map_frame, self._gt_base_frame, lookup_ts)

                    # Getting the map-odom and odom-base transforms independently because lookupTransform throws
                    # exceptions for the first poses
                    self._tf_listener.waitForTransform(self._map_frame, self._odom_frame, lookup_ts, timeout)
                    sl_mo_pose = self._tf_listener.lookupTransform(self._map_frame, self._odom_frame, lookup_ts)

                    self._tf_listener.waitForTransform(self._odom_frame, self._base_frame, lookup_ts, timeout)
                    sl_ob_pose = self._tf_listener.lookupTransform(self._odom_frame, self._base_frame, lookup_ts)

                except (tf.LookupException, tf.ConnectivityException,
                        tf.ExtrapolationException, tf2_ros.TransformException) as e:
                    rospy.loginfo("TF sequence {} exception: {}".format(seq, e))

                else:

                    pose_acq = True
                    break

        if seq_chgd and self._last_pose_acq:

            # Extract translation and rotation transforms from last poses
            gt_mb_t = np.array(self._last_gt_pose[0])
            gt_mb_r = np.array(self._last_gt_pose[1])
            sl_mo_t = np.array(self._last_sl_mo_pose[0])
            sl_mo_r = np.array(self._last_sl_mo_pose[1])
            sl_ob_t = np.array(self._last_sl_ob_pose[0])
            sl_ob_r = np.array(self._last_sl_ob_pose[1])

            # Convert to Homogeneous Transformation Matrices
            tf_ros = TransformerROS()
            gt_mb = tf_ros.fromTranslationRotation(gt_mb_t, gt_mb_r)
            sl_mo = tf_ros.fromTranslationRotation(sl_mo_t, sl_mo_r)
            sl_ob = tf_ros.fromTranslationRotation(sl_ob_t, sl_ob_r)

            # Multiply transforms from map-odom and odom-base to get map-base transform matrix
            sl_mb = np.dot(sl_ob, sl_mo)
            # Split into translation and rotation
            _, _, sl_mb_r, sl_mb_t, _ = decompose_matrix(sl_mb)
            # Convert rotation from Euler angles to quaternion and restructure the slam pose
            sl_mb_r = quaternion_from_euler(sl_mb_r[0], sl_mb_r[1], sl_mb_r[2])
            sl_pose = (sl_mb_t, sl_mb_r)
            sl_mb = tf_ros.fromTranslationRotation(sl_mb_t, sl_mb_r)

            # Compute relative pose between Ground Truth Base and SLAM Base poses
            rl_bb = np.dot(sl_mb, np.linalg.inv(gt_mb))
            _, _, rl_bb_r, rl_bb_t, _ = decompose_matrix(rl_bb)
            rl_bb_r = quaternion_from_euler(rl_bb_r[0], rl_bb_r[1], rl_bb_r[2])
            rel_pose = (rl_bb_t, rl_bb_r)

            # Translational Error as squared euclidean distance
            trans_error = np.matmul(rl_bb_t, rl_bb_t)
            self._cum_trans_err += trans_error

            # Rotational Error
            sl_mb_r_conj = quaternion_conjugate(sl_mb_r)
            q_diff = quaternion_multiply(gt_mb_r, sl_mb_r_conj)
            _, rot_error = quaternion_axis_angle(q_diff)
            _, rot_error2 = quaternion_axis_angle(rl_bb_r)
            # Normalize rotational error (-pi, pi)
            if rot_error > np.pi:
                rot_error -= 2 * np.pi
            rot_error = rot_error * rot_error
            self._cum_rot_err += rot_error

            tot_error = trans_error + self._lambda * rot_error
            self._cum_tot_err += tot_error

            if self._log_error:
                self._append_row(self._last_seq, self._last_ts, self._last_gt_pose, sl_pose, rel_pose,
                                 trans_error, rot_error, tot_error)

            if self._publish_error:
                err_msg = Float64()
                err_msg.data = self._cum_tot_err
                self._err_publisher.publish(err_msg)

        # Update last pose and info
        if seq_chgd:
            self._last_seq = seq

        if pose_acq:
            self._last_pose_acq = pose_acq
            self._last_ts = time_stamp

            self._last_gt_pose = gt_pose
            self._last_sl_mo_pose = sl_mo_pose
            self._last_sl_ob_pose = sl_ob_pose

    def _append_row(self, seq, ts, gt_pose, slam_pose, rel_pose, trans_err, rot_err, tot_err):
        """
        Append a row to the CSV file with the poses and errors

        :param seq: (int) Sequence number of the Ground Truth Pose message
        :param ts: (rospy.Time) Timestamp of the Ground Truth Pose message
        :param gt_pose: (tuple) Ground Truth Pose and orientation ([x, y, z], [x, y, z, w])
        :param slam_pose: (tuple) SLAM Pose and orientation ([x, y, z], [x, y, z, w])
        :param rel_pose: (tuple) Relative Pose and orientation ([x, y, z], [x, y, z, w])
        :param trans_err: (float) Step translational error
        :param rot_err: (float) Step rotational error
        :param tot_err: (float) Step total error

        :return: None
        """

        if not self._log_error:
            return

        gt_p_x, gt_p_y, gt_p_z = gt_pose[0]
        gt_r_x, gt_r_y, gt_r_z, gt_r_w = gt_pose[1]
        sl_p_x, sl_p_y, sl_p_z = slam_pose[0]
        sl_r_x, sl_r_y, sl_r_z, sl_r_w = slam_pose[1]
        rl_p_x, rl_p_y, rl_p_z = rel_pose[0]
        rl_r_x, rl_r_y, rl_r_z, rl_r_w = rel_pose[1]

        row = [
            seq, ts,
            gt_p_x, gt_p_y, gt_p_z, gt_r_x, gt_r_y, gt_r_z, gt_r_w,
            sl_p_x, sl_p_y, sl_p_z, sl_r_x, sl_r_y, sl_r_z, sl_r_w,
            rl_p_x, rl_p_y, rl_p_z, rl_r_x, rl_r_y, rl_r_z, rl_r_w,
            trans_err, self._cum_trans_err,
            rot_err, self._cum_rot_err,
            tot_err, self._cum_tot_err
        ]

        row_str = self._delim.join([str(x) for x in row])
        row_str += self._newline

        rospy.loginfo("Adding pose with seq. {}".format(seq))

        with open(self._log_file, 'a') as f:
            f.write(row_str)

    def _init_log(self):
        """
        Initialize the CSV log file by adding the column headers.

        :return: None
        """

        if not self._log_error:
            return

        rospy.loginfo("Saving error log to {}".format(self._log_file))

        col_head1 = [
            "SEQ", "Stamp",
            "GT", "", "", "", "", "", "",
            "SLAM", "", "", "", "", "", "",
            "REL", "", "", "", "", "", "",
            "ERROR", "", "", "", "", ""
        ]

        col_head2 = [
            "", "",
            "Pose", "", "", "Orientation", "", "", "",
            "Pose", "", "", "Orientation", "", "", "",
            "Pose", "", "", "Orientation", "", "", "",
            "Translational", "", "Angular", "",
            "Total (err_t + lambda * err_rot)[lambda = " + str(self._lambda) + "]", ""
        ]

        col_head3 = [
            "", "",
            "x", "y", "z", "x", "y", "z", "w",
            "x", "y", "z", "x", "y", "z", "w",
            "x", "y", "z", "x", "y", "z", "w",
            "Step", "Cumulative",
            "Step", "Cumulative",
            "Step", "Cumulative"
        ]

        col_headers = [col_head1, col_head2, col_head3]

        csv_header = self._newline.join([self._delim.join(col_header) for col_header in col_headers])
        csv_header += self._newline

        with open(self._log_file, 'w') as f:
            f.write(csv_header)
