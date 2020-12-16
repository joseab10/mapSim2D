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
    #move_options.remove(20)

    #rem_options = [path_file_options[2]] + path_file_options[4:]
    rem_options = path_file_options[3:]
    rem_combinations = itertools.product(*rem_options)

    curve_options = [path_file_options[1], path_file_options[2]]

    file_headers = ["Moves"] + map(str, move_options)
    file_cols = [file_headers]

    plt.ioff()

    for o1 in rem_combinations:

        fig_lbl = "_".join(o1)

        curves = []
        curve_combinations = itertools.product(*curve_options)

        for o2 in curve_combinations:

            col_lbl = "{}_{}_{}_{}_{}".format(o2[0], o2[1], o1[0], o1[1], o1[2])

            file_means = [col_lbl + "_mean"]
            file_sdevs = [col_lbl + "_var"]
            file_sems  = [col_lbl + "_sem"]
            file_runs  = [col_lbl + "_runs"]

            curve_lbl = "_".join(o2)
            #curve_lbl = str(o2)

            means = []
            sdevs = []
            sems = []
            runs = []

            for m in move_options:
                file_name = "{:03d}mv_{}.{}".format(m, col_lbl, args.extension)

                file_path = path.join(data_path, file_name)

                err = []
                no_experiments = 0

                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            err.append(float(line.split(FS)[-1]))
                            no_experiments += 1
                        except ValueError as e:
                            print("Warning: File {} contained unparsable line: {}.".format(file_name, repr(line)))

                means.append(np.mean(err))
                sdevs.append(np.std(err))
                sems.append(sem(err))
                runs.append(no_experiments)

            file_means += means
            file_sdevs += sdevs
            file_sems += sems
            file_runs += runs

            file_cols.append(map(str, file_means))
            file_cols.append(map(str, file_sdevs))
            file_cols.append(map(str, file_sems))
            file_cols.append(map(str, file_runs))

            curves.append((curve_lbl, means, sdevs))

        print("Plotting figure: " + fig_lbl)
        plt.figure()
        for curve in curves:
            plt.errorbar(move_options, curve[1], curve[2], label=curve[0])
        plt.legend()
        plt.xlabel("No. of mapping scans.")
        plt.ylabel("Error (mean +- stddev)")
        plt.xticks(move_options, rotation='vertical')
        plt.title(fig_lbl)
        plt.savefig(path.join(out_path, fig_lbl + "_errbar.svg"))
        plt.close()

    data_file_path = path.join(out_path, "errbar_data.dat")
    lines = zip(*file_cols)

    with open(data_file_path, "w") as f:
        for line in lines:
            f.write(FS.join(line) + LS)
