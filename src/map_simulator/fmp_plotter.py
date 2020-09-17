# ROS Libraries
import rospy

# ROS Messages
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from gmapping.msg import doubleMap, mapModel

# Math Libraries
import numpy as np
import numpy.ma as ma

import matplotlib
# Use non-interactive plotting back-end due to issues with rospy.spin()
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from cv_bridge import CvBridge

# OS Libraries
import os
import os.path
import datetime

# Project Libraries
# from map_simulator.map_models import MapModel
from map_simulator.map_colorizer import MapColorizer
from map_simulator.disc_states import DiscreteStates as DiSt


class FMPPlotter:
    """
    Class for plotting/coloring different statistics from the Full Map Posterior distribution
    and publishing them as images or saving them in files.
    """

    def __init__(self):
        """
        Constructor
        """

        rospy.init_node('fmp_plot')

        # Object for pseudo-coloring and plotting the maps
        self._map_colorizer = MapColorizer()

        self._alpha_map_sequence = -1
        self._beta_map_sequence = -2
        self._alpha_map = None
        self._beta_map = None

        self._img_seq = 1

        timestamp = datetime.datetime.now()
        timestamp = timestamp.strftime('%y%m%d_%H%M%S')
        self._path_timestamp = timestamp

        self._sub_topic_map_model = "map_model"
        self._sub_topic_fmp_alpha = "fmp_alpha"
        self._sub_topic_fmp_beta = "fmp_beta"

        self._map_model = None
        self._extent = [0, 100, 0, 100]

        # TODO: this two guys:
        # do_img_raw  = rospy.get_param("~img_raw" , False)
        # do_img_fmp  = rospy.get_param("~img_fmp" , False)
        do_img_stat = rospy.get_param("~img_stat", True)
        do_img_mlm  = rospy.get_param("~img_mlm" , False)
        do_img_para = rospy.get_param("~img_para", False)

        self._pub_img = rospy.get_param("~pub_img", True)
        self._topic_prefix = rospy.get_param("~pub_topic_prefix", "/fmp_img/")

        self._save_img = rospy.get_param("~save_img", True)
        self._resolution = rospy.get_param("~resolution", 300)
        default_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        default_path = os.path.join(default_path, 'FMP_img')
        save_dir = rospy.get_param("~save_dir", default_path)
        save_dir = os.path.expanduser(save_dir)
        save_dir = os.path.expandvars(save_dir)
        save_dir = os.path.normpath(save_dir)
        self._save_dir = save_dir

        self._alpha_prior = 1
        self._beta_prior = 0 if self._map_model == mapModel.DECAY_MODEL else 1

        # Image config dictionary
        sub_img_stat_mean_cfg = {"key": "mean", "dir": "stat", "file_prefix": "stat_mean",
                                 "topic": "stats/mean", "calc_f": self._calc_mean}
        sub_img_stat_var_cfg  = {"key": "var", "dir": "stat", "file_prefix": "stat_var",
                                 "topic": "stats/var", "calc_f": self._calc_var}
        img_stat_cfg = {"do": do_img_stat, "img": [sub_img_stat_mean_cfg, sub_img_stat_var_cfg]}

        sub_img_mlm_cfg = {"key": "mlm", "dir": "mlm", "file_prefix": "mlm",
                           "topic": "mlm", "calc_f": self._calc_mlm}

        img_mlm_cfg = {"do": do_img_mlm, "img": [sub_img_mlm_cfg]}

        sub_img_par_alpha_cfg = {"key": "alpha", "dir": "param", "file_prefix": "param_alpha",
                                 "topic": "param/alpha", "calc_f": self._calc_para_alpha}
        sub_img_par_beta_cfg = {"key": "beta", "dir": "param", "file_prefix": "param_beta",
                                "topic": "param/beta", "calc_f": self._calc_para_beta}
        img_par_cfg = {"do": do_img_para, "img": [sub_img_par_alpha_cfg, sub_img_par_beta_cfg]}

        self._img_cfg = {
            "stat": img_stat_cfg,
            "mlm": img_mlm_cfg,
            "par": img_par_cfg
        }

        fmp_param_sub_required = False

        # Create Publishers
        self._publishers = {}

        if self._pub_img:
            for img_set_key, img_set_cfg in self._img_cfg.items():
                fmp_param_sub_required = fmp_param_sub_required or img_set_cfg['do']
                if img_set_cfg['do']:
                    for img_cfg in img_set_cfg['img']:
                        topic = self._topic_prefix + img_cfg['topic']
                        self._publishers[img_cfg['key']] = rospy.Publisher(topic, Image,
                                                                           latch=True, queue_size=1)

        # Don't start the node if not needed...
        if (not self._pub_img and not self._save_img) or not fmp_param_sub_required:
            rospy.logerr('Nothing to do here! Why though?!?')
            return

        # Create Subscribers
        # To map model
        rospy.Subscriber(self._sub_topic_map_model, mapModel, self._map_model_callback)
        # To alpha and beta parameters (if publishing or saving images, and at least one image is generated)
        if (self._pub_img or self._save_img) and fmp_param_sub_required:
            rospy.Subscriber(self._sub_topic_fmp_alpha, doubleMap, self._map2d_alpha_callback)
            rospy.Subscriber(self._sub_topic_fmp_beta, doubleMap, self._map2d_beta_callback)

        # Create save path if not exists
        if self._save_img and fmp_param_sub_required:
            if not os.path.exists(self._save_dir):
                os.makedirs(self._save_dir)

        rospy.spin()

    def _fmp_acquired(self):
        """
        Checks if the map model has been received and if both alpha and beta maps have been received by comparing their
        sequence numbers.

        :return: True if map model, alpha and beta maps have been received, False otherwise.
        """

        return self._alpha_map_sequence == self._beta_map_sequence and self._map_model is not None

    def _plot(self):

        if not self._fmp_acquired():
            return

        if not self._pub_img and not self._save_img:
            return

        for img_set_key, img_set_cfg in self._img_cfg.items():
            if img_set_cfg['do']:
                rospy.loginfo('Plotting %s', img_set_key)
                for img_cfg in img_set_cfg['img']:

                    rospy.loginfo("\tComputing continuous and discrete images for %s.", img_cfg['key'])

                    alpha = self._alpha_prior + self._alpha_map
                    beta  = self._beta_prior  + self._beta_map

                    # Compute the images to plot using the configured calculation_function ('calc_f')
                    img_data = img_cfg['calc_f'](alpha, beta)

                    img_cont = img_data[0]  # Continuous valued map
                    img_disc = img_data[1]  # Discrete valued map
                    ds_list  = img_data[2]  # Discrete State list
                    v_min    = img_data[3]  # Minimum continuous value
                    v_max    = img_data[4]  # Maximum continuous value
                    occ      = img_data[5]  # Is occupancy map

                    self._map_colorizer.set_disc_state_list(ds_list)
                    self._map_colorizer.set_cont_bounds(img_cont, v_min=v_min, v_max=v_max, occupancy_map=occ)

                    rgba_img = self._map_colorizer.colorize(img_cont, img_disc, v_min, v_max)

                    if self._save_img:
                        path = os.path.join(self._save_dir, img_cfg['dir'])
                        path = os.path.join(path, self._path_timestamp)

                        if not os.path.exists(path):
                            os.makedirs(path)

                        filename = img_cfg['file_prefix'] + '_s' + str(self._alpha_map_sequence)
                        raw_filename = 'raw_' + filename + '.png'
                        filename = filename + '.svg'
                        mlp_path = os.path.join(path, filename)
                        raw_path = os.path.join(path, raw_filename)

                        fig, ax = plt.subplots(figsize=[20, 20])
                        ax.imshow(rgba_img, extent=self._extent)
                        self._map_colorizer.draw_cb_disc(fig)
                        if ds_list:
                            self._map_colorizer.draw_cb_cont(fig)

                        rospy.loginfo("\t\tSaving image %s to %s.", img_cfg['key'], mlp_path)
                        plt.savefig(mlp_path, bbox_inches='tight', dpi=self._resolution)
                        rospy.loginfo("\t\tSaving image %s to %s.", img_cfg['key'], raw_path)
                        plt.imsave(raw_path, rgba_img, vmin=0, vmax=1)

                        rospy.loginfo("\t\tImages saved.")

                    if self._pub_img:
                        pub_key = img_cfg['key']
                        publisher = self._publishers[pub_key]

                        rospy.loginfo("\t\tGenerating image message %s to %s.", img_cfg['key'], pub_key)

                        rgba_img = 255 * rgba_img
                        rgba_img = rgba_img.astype(np.uint8)

                        image_msg_head = Header()

                        image_msg_head.seq = self._img_seq
                        image_msg_head.stamp = rospy.Time.now()

                        br = CvBridge()
                        image_msg = br.cv2_to_imgmsg(rgba_img, encoding="rgba8")
                        image_msg.header = image_msg_head

                        publisher.publish(image_msg)

                        rospy.loginfo("\t\tImage published.")

        self._img_seq += 1

    def _map_reshape(self, msg):
        """
        Reshapes a map's data from a 1D list to a 2D ndarray.

        :param msg: (gmapping.doubleMap) A double precision floating point gmapping map message.

        :return: (ndarray) The map, reshaped as a 2D matrix.
        """

        w = msg.info.width
        h = msg.info.height

        reshaped_map = np.array(msg.data)
        reshaped_map = reshaped_map.reshape(w, h)
        reshaped_map = np.flipud(reshaped_map)

        # Set the plot's extension in world coordinates for meaningful plot ticks
        delta = msg.info.resolution
        x0 = msg.info.origin.position.x
        y0 = msg.info.origin.position.y
        x1 = x0 + w * delta
        y1 = y0 + h * delta

        self._extent = [x0, x1, y0, y1]
        self._map_colorizer.set_wm_extent(self._extent)

        return reshaped_map

    def _map_model_callback(self, msg):
        """
        Method called when receiving a map model type. It just sets the local field with the message's value.

        :param msg: (gmapping.mapModel) An integer stating the type of map model used by the SLAM algorithm and some
                                        constants for comparisons.

        :return: None
        """

        mm = msg.map_model
        mm_str = ''

        if mm == mapModel.REFLECTION_MODEL:
            mm_str = 'Reflection Model'
        elif mm == mapModel.DECAY_MODEL:
            mm_str = 'Exponential Decay Model'
        else:
            rospy.logerr('No idea what kind of model %d is! Going with Reflection Model.', mm)
            mm = mapModel.REFLECTION_MODEL

        rospy.loginfo("Received Map Model: (%d, %s)", mm, mm_str)

        self._map_model = mm

    def _map2d_alpha_callback(self, msg):
        """
        Method called when receiving a map with the alpha parameters of the full posterior map distribution.

        :param msg: (gmapping.doubleMap) A floating point gmapping map message.

        :return: None
        """

        self._alpha_map_sequence = msg.header.seq
        self._alpha_prior = msg.param

        self._alpha_map = self._map_reshape(msg)

        self._plot()

    def _map2d_beta_callback(self, msg):
        """
        Method called when receiving a map with the beta parameters of the full posterior map distribution.

        :param msg: (gmapping.doubleMap) A floating point gmapping map message.

        :return: None
        """

        self._beta_map_sequence = msg.header.seq
        self._beta_prior = msg.param

        self._beta_map = self._map_reshape(msg)

        self._plot()

    def _calc_mean(self, alpha, beta):
        shape = alpha.shape

        v_min = 0

        if self._map_model == mapModel.DECAY_MODEL:
            denominator = beta
            v_max = None

        elif self._map_model == mapModel.REFLECTION_MODEL:
            denominator = alpha + beta
            v_max = 1

        else:
            denominator = ma.ones(shape)
            v_max = None
            rospy.logerr('No valid map model defined!')

        undef_mask = (denominator == 0)

        numerator = ma.masked_array(alpha)
        numerator[undef_mask] = ma.masked

        means = ma.divide(numerator, denominator)

        means_ds = ma.zeros(shape)
        means_ds[undef_mask] = DiSt.UNDEFINED.value
        means_ds[~undef_mask] = ma.masked

        ds_list = [DiSt.UNDEFINED]
        occ = True

        return means, means_ds, ds_list, v_min, v_max, occ

    def _calc_var(self, alpha, beta):
        shape = alpha.shape

        v_min = 0
        v_max = None

        if self._map_model == mapModel.DECAY_MODEL:
            numerator = alpha
            denominator = np.multiply(beta, beta)
            # return np.divide(alpha, np.multiply(beta, beta), out=-1 * np.ones_like(alpha), when=(beta != 0))

        elif self._map_model == mapModel.REFLECTION_MODEL:
            a_plus_b = alpha + beta
            numerator = np.multiply(alpha, beta)
            denominator = np.multiply(np.multiply(a_plus_b, a_plus_b), (a_plus_b + 1))

        else:
            numerator = ma.ones(shape)
            denominator = ma.ones(shape)
            rospy.logerr('No valid map model defined!')

        undef_mask = (denominator == 0)

        numerator = ma.masked_array(numerator)
        numerator[undef_mask] = ma.masked

        variances = ma.divide(numerator, denominator)

        vars_ds = ma.zeros(shape)
        vars_ds[undef_mask] = DiSt.UNDEFINED.value
        vars_ds[~undef_mask] = ma.masked

        ds_list = [DiSt.UNDEFINED]
        occ = False

        return variances, vars_ds, ds_list, v_min, v_max, occ

    def _calc_mlm(self, alpha, beta):
        shape = alpha.shape

        numerator = ma.masked_array(alpha - 1)

        v_min = 0

        if self._map_model == mapModel.REFLECTION_MODEL:
            denominator = alpha + beta - 2

            undef_mask = (denominator == 0)
            n_undef_mask = ~undef_mask

            unif_mask = np.logical_and(alpha == 1, beta == 1)
            unif_mask = np.logical_and(unif_mask, n_undef_mask)
            bimod_mask = np.logical_and(alpha < 1, beta < 1)
            bimod_mask = np.logical_and(bimod_mask, n_undef_mask)

            mask = np.logical_or(undef_mask, unif_mask)
            mask = np.logical_or(mask, bimod_mask)

            numerator[mask] = ma.masked

            mlm = ma.divide(numerator, denominator)

            mlm_ds = ma.zeros(shape)
            mlm_ds[~mask] = ma.masked
            mlm_ds[unif_mask] = DiSt.UNIFORM.value
            mlm_ds[undef_mask] = DiSt.UNDEFINED.value
            mlm_ds[bimod_mask] = DiSt.BIMODAL.value

            ds_list = [DiSt.UNDEFINED, DiSt.UNIFORM, DiSt.BIMODAL]

            v_max = 1

        elif self._map_model == mapModel.DECAY_MODEL:
            denominator = beta

            undef_mask = np.logical_or(denominator == 0, alpha < 1)

            numerator[undef_mask] = ma.masked

            mlm = ma.divide(numerator, denominator)

            mlm_ds = ma.zeros(shape)
            mlm_ds[undef_mask] = DiSt.UNDEFINED.value
            mlm_ds[~undef_mask] = ma.masked

            ds_list = [DiSt.UNDEFINED]

            v_max = None

        else:
            rospy.logerr('No valid map model defined!')
            mlm = ma.zeros(shape)
            mlm_ds = None
            ds_list = []
            v_max = 1

        occ = True

        return mlm, mlm_ds, ds_list, v_min, v_max, occ

    @staticmethod
    def _calc_para_alpha(alpha, _):
        return alpha, None, [], 0, None, False

    @staticmethod
    def _calc_para_beta(_, beta):
        return beta, None, [], 0, None, False
