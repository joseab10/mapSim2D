from std_msgs.msg import Bool as BoolMsg

from misc_cmd import MiscCommand


class LocalizationMessageCommand(MiscCommand):
    """
    Class for sending a boolean message to the SLAM node to let it know to start the localization only phase, so as to
    evaluate the performance of the algorithm.
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate Localization-Only message command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "enable": (bool) (Optional)[Default: True) Send the localization-only
                                                     message if True.
                                  * "print": (bool) (Optional)[Default: True) Print a text during simulation if True to
                                                    let the user know that the message was sent.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command. Unused.
        """

        super(LocalizationMessageCommand, self).__init__(config, callback, last_pose)

        if 'enable' in config:
            self.en = bool(config['enable'])
        else:
            self.en = True

        if 'print' in config:
            self.do_print = bool(config['print'])
        else:
            self.do_print = True

        msg = BoolMsg()
        msg.data = self.en

        self.msg = msg
