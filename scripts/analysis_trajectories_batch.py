#!/usr/bin/env python2
# coding: utf-8

import os
import sys
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../src/rpg_trajectory_evaluation'))
from trajectory_v2 import TrajectoryV2

draw_color_list = ['lightcoral','orange','lightskyblue','lightslategrey','royalblue','plum','pink']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def plot_abs_errors(dataset_errors, save_figure=True, output_dir=''):
    print ("###### Plot_abs_errors... #######")
    dataset_list = list(dataset_errors.keys())
    assert len(dataset_errors) > 0, "No Dataset!!! Nothing to plot!!!"
    dataset_list.sort()

    algorithm_list = list(dataset_errors[dataset_list[0]].keys())
    assert len(algorithm_list) > 0, "No Algorithm!!! Nothing to plot!!!"
    algorithm_list.sort()

    assert len(draw_color_list) >= len(algorithm_list), "No Enough Colours for the Algorithms!!!"

    # prepare the space of the plots
    x = np.arange(len(dataset_list))
    width_per_dataset = 0.8
    width_per_algo = width_per_dataset / len(algorithm_list)
    bar_width = width_per_algo * 0.75

    ########## plot RMSE
    plt.figure()
    # subplot: translation
    plt.subplot(221)
    algo_trans_rmse_list = np.array([[dataset_errors[dataset][algo]['abs_errors']['abs_e_trans_stats']['rmse']
                                      for dataset in dataset_list]
                                     for algo in algorithm_list])

    for i in range(len(algorithm_list)):
        x_cur = x + np.array([i * width_per_algo for j in range(len(x))])
        plt.bar(x_cur, algo_trans_rmse_list[i], width=bar_width, label=algorithm_list[i], fc=draw_color_list[i])
    plt.xticks(x - width_per_algo / 2 + width_per_dataset / 2, dataset_list)
    plt.legend()
    plt.ylabel("RMSE-Translation (m)")

    # subplot: orientation
    plt.subplot(223)
    algo_rot_rmse_list = np.array([[dataset_errors[dataset][algo]['abs_errors']['abs_e_rot_stats']['rmse']
                                    for dataset in dataset_list]
                                   for algo in algorithm_list])
    for i in range(len(algorithm_list)):
        x_cur = x + np.array([i * width_per_algo for j in range(len(x))])
        plt.bar(x_cur, algo_rot_rmse_list[i], width=bar_width, label=algorithm_list[i], fc=draw_color_list[i])
    plt.xticks(x - width_per_algo / 2 + width_per_dataset / 2, dataset_list)
    plt.ylabel("RMSE-Orientation (deg)")

    ########## boxplot trajectory error statistics
    # subplot: translation
    plt.subplot(222)
    for i, algo in enumerate(algorithm_list):
        errors = []
        for dataset in dataset_list:
            errors.append(dataset_errors[dataset][algo]['abs_errors']['abs_e_trans'])
        x_cur = x + np.array([i * width_per_algo for j in range(len(x))])
        bp = plt.boxplot(errors, positions=x_cur, sym='', widths=bar_width, showmeans=True)
        set_box_color(bp, color=draw_color_list[i])
        plt.plot([], color=draw_color_list[i], label=algo)
    plt.legend()
    plt.xticks(x - width_per_algo / 2 + width_per_dataset / 2, dataset_list)
    plt.xlim(0 - bar_width, len(dataset_list) - 1 + width_per_dataset)
    plt.ylabel("ATE boxplot - Trans. (m)")

    # subplot: orientation
    plt.subplot(224)
    for i, algo in enumerate(algorithm_list):
        errors = []
        for dataset in dataset_list:
            errors.append(dataset_errors[dataset][algo]['abs_errors']['abs_e_rot'])
        x_cur = x + np.array([i * width_per_algo for j in range(len(x))])
        bp = plt.boxplot(errors, positions=x_cur, sym='', widths=bar_width, showmeans=True)
        set_box_color(bp, color=draw_color_list[i])
        plt.plot([], color=draw_color_list[i], label=algo)
    plt.xticks(x - width_per_algo / 2 + width_per_dataset / 2, dataset_list)
    plt.xlim(0 - bar_width, len(dataset_list) - 1 + width_per_dataset)
    plt.ylabel("ATE boxplot - Ori. (deg)")
    plt.suptitle("Absolute Trajectory Error", color='lightslategrey', fontsize=16, weight='bold')

    if save_figure:
        fig = plt.gcf()
        fig.set_size_inches((20, 10), forward=False)
        plt.savefig(output_dir + 'ate_all_datasets.png', dpi=300)

    print ("###### ......Finished! #######")

def plot_travel_errors(dataset_errors, save_figure=True, output_dir=''):
    print ("###### plot_travel_errors... #######")
    dataset_list = list(dataset_errors.keys())
    assert len(dataset_errors) > 0, "No Dataset!!! Nothing to plot!!!"
    dataset_list.sort()

    algorithm_list = list(dataset_errors[dataset_list[0]].keys())
    assert len(algorithm_list) > 0, "No Algorithm!!! Nothing to plot!!!"
    algorithm_list.sort()

    assert len(draw_color_list) >= len(algorithm_list), "No Enough Colours for the Algorithms!!!"
    for i, dataset in enumerate(dataset_list):
        print ("...processing " + dataset)
        plt.figure()
        plt.subplot(311)
        for j, algo in enumerate(algorithm_list):
            err = dataset_errors[dataset][algo]['abs_errors']['abs_e_trans']
            # index = np.arange(len(err))
            index = dataset_errors[dataset][algo]['accum_errors']['accum_distance']
            plt.plot(index, err, label=algo, color=draw_color_list[j])
        plt.legend()
        plt.ylabel("Per Pose Error - Trans. (m)")

        plt.subplot(312)
        for j, algo in enumerate(algorithm_list):
            err = dataset_errors[dataset][algo]['abs_errors']['abs_e_rot']
            # index = np.arange(len(err))
            index = dataset_errors[dataset][algo]['accum_errors']['accum_distance']
            plt.plot(index, err, label=algo, color=draw_color_list[j])
        plt.ylabel("Per Pose Error - Ori. (deg)")

        plt.subplot(313)
        for j, algo in enumerate(algorithm_list):
            err = dataset_errors[dataset][algo]['accum_errors']['accum_distance_error']
            accum_error_ratio = dataset_errors[dataset][algo]['accum_errors']['accum_distance_error_ratio']
            # index = np.arange(len(err))
            index = dataset_errors[dataset][algo]['accum_errors']['accum_distance']
            label = algo + ": travel distance error ratio " + str(round(accum_error_ratio * 100, 2)) + "%"
            plt.plot(index, err, label=label, color=draw_color_list[j])
        plt.legend()
        plt.xlabel("Travelled Distance (m)",fontsize=12)
        plt.ylabel("Accumulated Distance Error (m)")
        plt.suptitle("Whole Trajectory Errors: " + dataset, color='lightslategrey', fontsize=16, weight='bold')
        if save_figure:
            fig = plt.gcf()
            fig.set_size_inches((20, 10), forward=False)
            plt.savefig(output_dir + "whole_traj_error_" + dataset + ".png", dpi=300)

    print ("###### ......Finished! #######")

def plot_rel_errors(dataset_errors, save_figure=True, output_dir=''):
    print ("###### plot_rel_errors... #######")
    dataset_list = list(dataset_errors.keys())
    assert len(dataset_errors) > 0, "No Dataset!!! Nothing to plot!!!"
    dataset_list.sort()

    algorithm_list = list(dataset_errors[dataset_list[0]].keys())
    assert len(algorithm_list) > 0, "No Algorithm!!! Nothing to plot!!!"
    algorithm_list.sort()

    assert len(draw_color_list) >= len(algorithm_list), "No Enough Colours for the Algorithms!!!"

    ########## plot relative errors as a per dataset manner
    for i, dataset in enumerate(dataset_list):
        print("...processing " + dataset)
        # prepare the space of the plots
        subtraj_length = list(dataset_errors[dataset][algorithm_list[0]]['rel_errors'].keys())
        subtraj_length.sort()
        x = np.arange(len(subtraj_length))
        width_per_sublen = 0.8
        width_per_algo = width_per_sublen / len(algorithm_list)
        bar_width = width_per_algo * 0.75

        plt.figure("Relative Errors: " + dataset)
        plt.subplot(221)
        rmse_list = np.array([[dataset_errors[dataset][algo]['rel_errors'][sub_len]['rel_trans_stats']['rmse']
                               for sub_len in subtraj_length]
                              for algo in algorithm_list])

        for i in range(len(algorithm_list)):
            x_cur = x + np.array([i * width_per_algo for j in range(len(x))])
            plt.bar(x_cur, rmse_list[i], width=bar_width, label=algorithm_list[i], fc=draw_color_list[i])
        plt.xticks(x - width_per_algo / 2 + width_per_sublen / 2, subtraj_length)
        plt.legend()
        plt.xlabel("Sub-trajectory length (m)")
        plt.ylabel("RMSE-Translation (m)")

        plt.subplot(223)
        rmse_list = np.array([[dataset_errors[dataset][algo]['rel_errors'][sub_len]['rel_rot_stats']['rmse']
                               for sub_len in subtraj_length]
                              for algo in algorithm_list])
        for i in range(len(algorithm_list)):
            x_cur = x + np.array([i * width_per_algo for j in range(len(x))])
            plt.bar(x_cur, rmse_list[i], width=bar_width, label=algorithm_list[i], fc=draw_color_list[i])
        plt.xticks(x - width_per_algo / 2 + width_per_sublen / 2, subtraj_length)
        # plt.legend()
        plt.xlabel("Sub-trajectory length (m)")
        plt.ylabel("RMSE-Orientation (deg)")

        plt.subplot(222)
        for i, algo in enumerate(algorithm_list):
            errors = []
            for sub_len in subtraj_length:
                errors.append(dataset_errors[dataset][algo]['rel_errors'][sub_len]['rel_trans'])
            x_cur = x + np.array([i * width_per_algo for j in range(len(x))])
            bp = plt.boxplot(errors, positions=x_cur, sym='', widths=bar_width, showmeans=True)
            set_box_color(bp, color=draw_color_list[i])
            plt.plot([], color=draw_color_list[i], label=algo)
        plt.legend()
        plt.xticks(x - width_per_algo / 2 + width_per_sublen / 2, subtraj_length)
        plt.xlim(0 - bar_width, len(subtraj_length) - 1 + width_per_sublen)
        plt.xlabel("Sub-trajectory length (m)")
        plt.ylabel("RPE boxplot Trans. (m)")

        plt.subplot(224)
        for i, algo in enumerate(algorithm_list):
            errors = []
            for sub_len in subtraj_length:
                errors.append(dataset_errors[dataset][algo]['rel_errors'][sub_len]['rel_rot'])
            x_cur = x + np.array([i * width_per_algo for j in range(len(x))])
            bp = plt.boxplot(errors, positions=x_cur, sym='', widths=bar_width, showmeans=True)
            set_box_color(bp, color=draw_color_list[i])
            plt.plot([], color=draw_color_list[i], label=algo)
        # plt.legend()
        plt.xticks(x - width_per_algo / 2 + width_per_sublen / 2, subtraj_length)
        plt.xlim(0 - bar_width, len(subtraj_length) - 1 + width_per_sublen)
        plt.xlabel("Sub-trajectory length (m)")
        plt.ylabel("RPE boxplot Ori. (deg)")

        plt.suptitle("Relative Pose Error: " + dataset, color='lightslategrey', fontsize=16, weight='bold')
        if save_figure:
            fig = plt.gcf()
            fig.set_size_inches((20, 10), forward=False)
            plt.savefig(output_dir + "rpe_" + dataset + ".png", dpi=300)

    print ("###### ......Finished! #######")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='''Analysis Trajectories''')
    parser.add_argument("result_folder",    type=str,   help="the folder with the given structure")
    parser.add_argument("--align_mode",     type=str,   help="the alignment mode of two trajectories",
                        default="sim3", choices=["posyaw","se3","sim3","none"])
    parser.add_argument("--save_figure",    type=bool,  help="whether save the result figure or not", default=True)
    parser.add_argument("--output_folder",  type=str,   help="the location to save the figure", default="")

    args = parser.parse_args()
    assert os.path.exists(args.result_folder), "gt_folder not exist!!!"
    assert len(os.listdir(args.result_folder)) > 0, "gt_folder EMPTY!!!"

    # dataset_errors = np.load('../results/samples/whole_errors_default_subtraj.npy', allow_pickle=True).item()
    # plot_abs_errors(dataset_errors, output_dir=args.output_folder)
    # plot_travel_errors(dataset_errors, output_dir=args.output_folder)
    # plot_rel_errors(dataset_errors, output_dir=args.output_folder)
    # exit(0)

    datasets = [name for name in os.listdir(args.result_folder) if os.path.isdir(os.path.join(args.result_folder, name))]
    datasets.sort()
    dataset_errors = {} # store errors per dataset
    for dataset in datasets: # for each dataset
        dataset_path = os.path.join(args.result_folder + '/' + dataset)
        cur_gt_file = os.path.join(dataset_path + '/' + dataset + ".txt")
        if os.path.exists(cur_gt_file) == False: # check the existence of the groundtruth trajectory
            warnings.warn("groundtruth of dataset {0} not exist!!!".format(dataset))
            continue
        algorithms = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
        if len(algorithms) == 0:
            warnings.warn("NO algorithm in the dataset {0} folder".format(dataset))
            continue
        algorithms.sort()
        algo_errors = {} # store errors per algorithm
        for algo in algorithms: # for each algorithm in the dataset folder
            print("=====Processing Dataset {0} with Algorithm {1}...=====".format(dataset, algo))
            algo_path = os.path.join(dataset_path + '/' + algo)
            algo_runs = os.listdir(algo_path)
            if len(algo_runs) == 0:
                warnings.warn("The algorithm {0} does NOT provide estimations of dataset {1}!!!".format(algo, dataset))
                continue
            cur_est_file = os.path.join(algo_path + '/' + algo_runs[0])
            cur_traj = TrajectoryV2(cur_gt_file, cur_est_file, 'posyaw')
            cur_traj.compute_absolute_error()
            cur_traj.compute_accumulate_errors()
            # subtraj_lengths = [8.0, 16.0, 24.0, 32.0, 40.0]
            cur_traj.compute_relative_errors()
            single_errors = {} # store errors per trajectory
            single_errors['abs_errors'] = cur_traj.abs_errors
            single_errors['rel_errors'] = cur_traj.rel_errors
            single_errors['accum_errors'] = cur_traj.accum_errors
            algo_errors[algo] = single_errors
            print("=====...Finished=====")
            print("\n")

        # structure: datasets->algorithms->errors
        dataset_errors[dataset] = algo_errors
    # may cache the trajectories results;
    # np.save('whole_errors_default_subtraj.npy', dataset_errors)
    plot_abs_errors(dataset_errors, output_dir=args.output_folder)
    plot_travel_errors(dataset_errors, output_dir=args.output_folder)
    plot_rel_errors(dataset_errors, output_dir=args.output_folder)



