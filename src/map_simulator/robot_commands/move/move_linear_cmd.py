import numpy as np

from move_interpol_cmd import MoveInterpolationCommand

from map_simulator.utils import to_np


class MoveLinearCommand(MoveInterpolationCommand):
    """
    Class for a command for moving from a given start pose, to a given destination pose,
    in a given number of steps, by linearly interpolating the position's x and y coordinates
    between start and end, and having the robot's orientation follow the line defined by the
    start and end points.
    Works by computing the angle of the line between start and end positions, and feeding it
    to the Interpolation command in the configuration dictionary.
    """

    def __init__(self, config, callback, last_pose):
        """
        Instantiate a Linear move command object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "start_position": (list|np.ndarray) (Optional)[Default: last_pose.position)
                                                                        Starting position [x, y] of the robot.
                                  * "end_position": (list|np.ndarray) Ending position [x, y] of the robot.
                                  * "steps": (int) Number of desired steps/poses for the movement
        :param callback: (callable) Function, lambda or other callable object to be executed
                                    when calling the command object.
        :param last_pose: (Pose) Last pose of the robot before this command.
        """

        if 'start_position' in config:
            start = to_np(config['start_position'])
        else:
            start = last_pose.position

        end = to_np(config['end_position'])

        diff = end - start
        theta = np.arctan2(diff[1], diff[0])
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        config['end_orientation'] = theta

        if 'start_position' in config or 'start_orientation' in config or abs(theta - last_pose.orientation) > 1e-6:
            config['start_orientation'] = theta

        super(MoveLinearCommand, self).__init__(config, callback, last_pose)
