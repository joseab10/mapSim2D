#! /usr/bin/env python

import rospkg

import argparse

from os import path
from datetime import datetime

import itertools
from collections import OrderedDict

import multiprocessing

from map_simulator.ros_launcher import ROSLauncher


def run_exp_n_times(package, launch_file_path, iterations=1, launch_args_dict=None, log_path=None,
                    port=None, monitored_nodes=None):

    for i in range(iterations):
        # Run Launch file
        launch_args_dict["ts"] = datetime.now().strftime('%y%m%d_%H%M%S')

        launcher = ROSLauncher(package, launch_file_path, wait_for_master=False, log_path=log_path, port=port,
                               monitored_nodes=monitored_nodes)
        launcher.start(launch_args_dict)
        launcher.spin()


def int_list(string):
    return list(map(int, string.strip().split(',')))


if __name__ == "__main__":

    launchfile_package = "map_simulator"
    pck = rospkg.RosPack()
    launch_pck_share = pck.get_path(launchfile_package)
    def_launch_file = path.join(launch_pck_share, "launch", "experiment.launch")
    slamdata_pck_share = pck.get_path('slam_datasets')

    def_file_path = path.join("~", "Desktop", "Experiments", "MethodComparison")
    def_file_path = path.expanduser(def_file_path)
    def_file_path = path.expandvars(def_file_path)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run several mapping and localization experiments and collect data")
    parser.add_argument('-i', '--iterations', action='store', type=int, default=100,
                        help='Number of times to run each experiment.')
    parser.add_argument('-m', '--moves', action='store', type=int_list,
                        default="20,40,60,80,100,120,140,160,180,200,240,270,300",
                        help='Comma-separated list of number of movements to run the tests with.')
    parser.add_argument('-f', '--launch_file', action='store', type=str, default=def_launch_file,
                        help='Launch file to execute.')
    parser.add_argument('-w', '--num_workers', action='store', type=int, default=-1,
                        help='Number of workers/processes to run in parallel. (-1 to start one per core.')
    parser.add_argument('-p', '--path', action='store', type=str, default=def_file_path,
                        help='Launch file to execute.')

    args = parser.parse_args()

    iterations = args.iterations

    run_path = path.join(args.path, "run")
    log_path = path.join(args.path, "log")
    err_path = path.join(args.path, "err")
    launch_file = args.launch_file

    # Static Launch Arguments (Don't change between experiments)
    stat_args = {
        "do_slam": True,
        "do_plots": False,
        "do_error": True,
        "do_coll_err": True,
        "do_gtmap": False,
        "do_odo": False,
        "do_rviz": False,
        "do_gsp_debug": True,
        "sim_pause": False,
        "sim_quiet": True,

        "path_err_coll_path": err_path,
        "path_prefix": run_path
    }

    # Variable Launch Arguments (Different for each experiment)
    # All permutations of these settings will be executed, so be wary of that!
    var_args = OrderedDict([
        #("bag_file", [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 270, 300]),
        #("bag_file", [20, 40, 60, 80, 100]),
        ("bag_file", args.moves),
        ("mm", ["ref", "dec"]),
        ("pw", ["cmh", "ml"]),
        ("doPoseImprove", [True, False])
    ])

    # Functions for transforming an argument into its final value and label
    bag_file_format = "Robot_Exp_10Loop_{:03d}m1loc.bag"
    bag_file_format = path.join(slamdata_pck_share, "Simulations", bag_file_format)
    var_args_fun = {
        "val": {
            "bag_file": lambda x: bag_file_format.format(x)
        },
        "lbl": {
            "bag_file": lambda x: "{:03d}mv".format(x),
            "doPoseImprove": lambda x: "sm" if x else "mm"
        }
    }

    # Generate all possible experiment permutations
    keys, values = zip(*var_args.items())
    var_args = [OrderedDict(zip(keys, v)) for v in itertools.product(*values)]

    # Generating final list of arguments by executing their transformation functions if needed.
    experiment_arg_list = []
    for exp_args in var_args:
        experiment_args = {}
        file_prefix = ""
        for k, v in exp_args.items():
            if k in var_args_fun["lbl"]:
                file_prefix += var_args_fun["lbl"][k](v) + "_"
            else:
                file_prefix += str(v) + "_"
            if k in var_args_fun["val"]:
                experiment_args[k] = var_args_fun["val"][k](v)
            else:
                experiment_args[k] = v
        file_prefix = file_prefix[:-1]
        experiment_args["path_err_coll_pfx"] = file_prefix

        launch_args_dict = stat_args.copy()
        launch_args_dict.update(experiment_args)
        experiment_arg_list.append(launch_args_dict)

    # Multiprocess pool settings
    if args.num_workers < 1:
        num_procs = multiprocessing.cpu_count()-1 or 1
    else:
        num_procs = args.num_workers

    multiproc = True
    if multiproc:
        pool = multiprocessing.Pool(processes=num_procs)
        print("Running experiments in a pool of {} processes.".format(num_procs))
        procs = []

    # Main experiments
    for exp_args in experiment_arg_list:
        if not multiproc:
            run_exp_n_times(launchfile_package, launch_file, iterations=iterations, launch_args_dict=exp_args,
                            log_path=log_path, monitored_nodes={"any": ["sim", "SLAM/slam"]}, port="auto")
        else:
            proc = pool.apply_async(run_exp_n_times, args=(launchfile_package, launch_file, ),
                                kwds={"iterations": iterations,
                                      "launch_args_dict": exp_args,
                                      "log_path": log_path,
                                      "monitored_nodes": {"all": ["sim", "SLAM/slam"]},
                                      "port": "auto"}
                                )
            procs.append(proc)

    if multiproc:
        for i, proc in enumerate(procs):
            print("\n\n\nProcess {}".format(i))
            print(proc.get())
