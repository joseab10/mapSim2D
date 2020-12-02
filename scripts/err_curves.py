#! /usr/bin/env python

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

import itertools

from os import path, listdir, makedirs

if __name__ == "__main__":

    import argparse

    FS = ","
    LS = "\n"

    def_data_path = path.join("~", "Desktop", "Experiments", "MethodComparison", "err")
    def_out_path = path.join(def_data_path, "errbar")

    parser = argparse.ArgumentParser(description='Read the data collected into csv files in a given directory \
        and plot error bars.')
    parser.add_argument('-d', '--dir', action='store', type=str, default=def_data_path,
                        help='Path of the directory where the CSV error files are stored.')
    parser.add_argument('-x', '--extension', action='store', type=str, default='csv',
                        help='Data file extension. [Default: csv].')
    parser.add_argument('-o', '--out_dir', action='store', type=str, default=def_out_path,
                        help='Output Directory where histograms will be saved.')

    args = parser.parse_args()
    data_path = path.expandvars(path.expanduser(args.dir))
    out_path = path.expandvars(path.expanduser(args.out_dir))

    if not path.exists(out_path):
        makedirs(out_path)

    tmp_ext = ".{}".format(args.extension)
    path_files = listdir(data_path)
    path_files = [f[:f.find(tmp_ext)] for f in path_files if tmp_ext in f]

    path_file_options = [f.split("_") for f in path_files]
    path_file_options = (zip(*path_file_options))
    path_file_options = [sorted(list(set(c))) for c in path_file_options]

    move_options = sorted(map(int, [o[:o.find('mv')] for o in path_file_options[0]]))

    rem_options = [path_file_options[2]] + path_file_options[4:]
    rem_options = itertools.product(*rem_options)

    file_headers = ["Moves"] + map(str, move_options)
    file_cols = [file_headers]

    plt.ioff()

    for o1 in rem_options:

        fig_lbl = "{}_{}_{}".format(o1[0], o1[1], o1[2])

        curves = []

        curve_options = [path_file_options[1], path_file_options[3]]
        curve_options = itertools.product(*curve_options)

        for o2 in curve_options:

            col_lbl = "{}_{}_{}_{}_{}".format(o2[0], o1[0], o2[1], o1[1], o1[2])

            file_means = [col_lbl + "_mean"]
            file_vars  = [col_lbl + "_var"]
            file_sems  = [col_lbl + "_sem"]
            file_runs  = [col_lbl + "_runs"]

            curve_lbl = "{}_{}".format(o2[0], o2[1])

            means = []
            varis = []
            sems = []
            runs = []

            for m in move_options:
                file_name = "{:03d}mv_{}_{}_{}_{}_{}.{}".format(m, o2[0], o1[0], o2[1], o1[1], o1[2], args.extension)

                file_path = path.join(data_path, file_name)

                err = []
                no_experiments = 0

                with open(file_path, 'r') as f:
                    for line in f:
                        err.append(float(line.split(FS)[-1]))
                        no_experiments += 1

                means.append(np.mean(err))
                varis.append(np.var(err))
                sems.append(sem(err))
                runs.append(no_experiments)

            file_means += means
            file_vars += varis
            file_sems += sems
            file_runs += runs

            file_cols.append(map(str, file_means))
            file_cols.append(map(str, file_vars))
            file_cols.append(map(str, file_sems))
            file_cols.append(map(str, file_runs))

            curves.append((curve_lbl, means, sems))

        print("Plotting figure: " + fig_lbl)
        plt.figure()
        for curve in curves:
            plt.errorbar(move_options, curve[1], curve[2], label=curve[0])
        plt.legend()
        plt.title(fig_lbl)
        plt.savefig(path.join(out_path, fig_lbl + "_errbar.svg"))
        plt.close()

    data_file_path = path.join(out_path, "errbar_data.dat")
    lines = zip(*file_cols)


    with open(data_file_path, "w") as f:
        for line in lines:
            f.write(FS.join(line) + LS)









    # curves = [
    #     {"lbl": "ref", "file_sfx": "ref_cmh_"},
    #     {"lbl": "ref_FMP_ML", "file_sfx": "ref_ml_"}
    # ]
    # tests = ["map", "loc"]
    # err = ["tra", "rot", "tot"]
    #
    # np.array()
    # moves = ["{:03d}mv".format(m) for m in moves]
    #
    # for file in files:
    #
    #     file_path = path.join(data_path, file)
    #
    #     for err_type in err_types:
    #
    #         fig_name = file + "_" + err_type
    #         print("Plotting figure: " + fig_name)
    #         plt.figure()
    #
    #         for test_env in test_envs:
    #
    #             err_file_path = file_path + '_' + test_env + '_' + err_type + '.' + extension
    #             err = []
    #             with open(err_file_path, 'r') as f:
    #                 for line in f:
    #                     err.append(float(line.split(FS)[-1]))
    #
    #             plt.hist(err, label=test_env)