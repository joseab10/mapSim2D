from misc_cmd import MiscCommand


class ScanCommand(MiscCommand):
    """
    Class for initiating a scan of the environment on command, apart from the automatic ones done after
    the move commands.
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate Scan command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "scans": (int) (Optional)[Default: 1) Number of scans to perform.
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command. Unused.
        """

        super(ScanCommand, self).__init__(config, callback, last_pose)

        if 'scans' in config:
            self.scans = int(config['scans'])
        else:
            self.scans = 1
