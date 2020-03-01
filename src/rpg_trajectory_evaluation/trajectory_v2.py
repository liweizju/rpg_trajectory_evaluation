#!/usr/bin/env python2

import os
import yaml
import pickle
import time

import numpy as np

import trajectory_utils as traj_utils
import results_writer as res_writer
import compute_trajectory_errors as traj_err
import align_utils as au
import associate_timestamps

import transformations as tf


class TrajectoryV2:

    def __init__(self, gt_traj_file, est_traj_file, align_type='sim3', align_num_frames=-1,
                 time_offset=0.0, max_diff=0.02):

        assert os.path.exists(gt_traj_file),\
            "Specified groundtruth trajectory file {0} does not exist.".format(gt_traj_file)
        assert os.path.exists(est_traj_file),\
            "Specified estimation trajectory file {0} does not exist.".format(est_traj_file)
        assert align_type in ['posyaw', 'se3', 'sim3', 'none']

        self.gt_traj_file = gt_traj_file
        self.est_traj_file = est_traj_file

        self.time_offset = time_offset
        self.max_diff = max_diff

        self.align_type = align_type
        self.align_num_frames = align_num_frames

        self.data_aligned = False
        self.data_loaded = False

        self.abs_errors = {}
        self.rel_errors = {}
        self.accum_errors = {}

        self.load_data()
        self.align_trajectory()

    def load_data(self):
        """
        Loads the trajectory data. The resuls {p_es, q_es, p_gt, q_gt} is
        synchronized and has the same length.
        """
        if self.data_loaded == True:
            print("Data already loaded")
            return
        print('Loading trajectory data...')
        # associate the estimation and groundtruth
        matches = associate_timestamps.read_files_and_associate(
            self.est_traj_file, self.gt_traj_file, self.time_offset, self.max_diff)
        dict_matches = dict(matches)

        data_es = np.loadtxt(self.est_traj_file)
        data_gt = np.loadtxt(self.gt_traj_file)

        self.p_es = []
        self.p_gt = []
        self.q_es = []
        self.q_gt = []
        self.t_gt = []
        for es_id, es in enumerate(data_es):
            if es_id in dict_matches:
                gt = data_gt[dict_matches[es_id]]
                self.p_es.append(es[1:4])
                self.p_gt.append(gt[1:4])
                self.q_es.append(es[4:8])
                self.q_gt.append(gt[4:8])
                self.t_gt.append(gt[0])
        self.p_es = np.array(self.p_es)
        self.p_gt = np.array(self.p_gt)
        self.q_es = np.array(self.q_es)
        self.q_gt = np.array(self.q_gt)
        self.t_gt = np.array(self.t_gt)

        self.accum_distances = traj_utils.get_distance_from_start(self.p_gt)
        self.traj_length = self.accum_distances[-1]

        self.data_loaded = True;
        print('...done.')


    def align_trajectory(self):
        if self.data_aligned:
            print("Trajectory already aligned")
            return
        print("Aliging the trajectory estimate to the groundtruth...")

        print("Alignment type is {0}.".format(self.align_type))
        n = int(self.align_num_frames)
        if n < 0.0:
            print('To align all frames.')
            n = len(self.p_es)
        else:
            print('To align trajectory using ' + str(n) + ' frames.')

        self.trans = np.zeros((3,))
        self.rot = np.eye(3)
        self.scale = 1.0
        if self.align_type == 'none':
            pass
        else:
            self.scale, self.rot, self.trans = au.alignTrajectory(
                self.p_es, self.p_gt, self.q_es, self.q_gt, 
                self.align_type, self.align_num_frames)

        self.p_es_aligned = np.zeros(np.shape(self.p_es))
        self.q_es_aligned = np.zeros(np.shape(self.q_es))
        for i in range(np.shape(self.p_es)[0]):
            self.p_es_aligned[i, :] = self.scale * \
                self.rot.dot(self.p_es[i, :]) + self.trans
            q_es_R = self.rot.dot(
                tf.quaternion_matrix(self.q_es[i, :])[0:3, 0:3])
            q_es_T = np.identity(4)
            q_es_T[0:3, 0:3] = q_es_R
            self.q_es_aligned[i, :] = tf.quaternion_from_matrix(q_es_T)

        self.data_aligned = True
        print("... trajectory alignment done.")

    def compute_absolute_error(self):
        if self.abs_errors:
            print("Absolute errors already calculated")
        else:
            print('Calculating RMSE...')

            e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc =\
                traj_err.compute_absolute_error(self.p_es_aligned,
                                                self.q_es_aligned,
                                                self.p_gt,
                                                self.q_gt)
            stats_trans = res_writer.compute_statistics(e_trans)
            stats_rot = res_writer.compute_statistics(e_rot)
            # stats_scale = res_writer.compute_statistics(e_scale_perc)

            self.abs_errors['abs_e_trans'] = e_trans
            self.abs_errors['abs_e_trans_stats'] = stats_trans
            # self.abs_errors['abs_e_trans_vec'] = e_trans_vec
            self.abs_errors['abs_e_rot'] = e_rot
            self.abs_errors['abs_e_rot_stats'] = stats_rot
            # self.abs_errors['abs_e_ypr'] = e_ypr
            # self.abs_errors['abs_e_scale_perc'] = e_scale_perc
            # self.abs_errors['abs_e_scale_stats'] = stats_scale
            print('...RMSE calculated.')
        return

    def compute_relative_error_at_subtraj_len(self, subtraj_len,
                                              max_dist_diff=-1):
        if max_dist_diff < 0:
            max_dist_diff = 0.2 * subtraj_len

        if self.rel_errors and (subtraj_len in self.rel_errors):
            print("Relative error at sub-trajectory length {0} is already "
                  "computed or loaded from cache.".format(subtraj_len))
        else:
            print("Computing relative error at sub-trajectory "
                  "length {0}".format(subtraj_len))
            Tcm = np.identity(4)
            _, e_trans, e_trans_perc, e_yaw, e_gravity, e_rot =\
                traj_err.compute_relative_error(
                    self.p_es, self.q_es, self.p_gt, self.q_gt, Tcm,
                    subtraj_len, max_dist_diff, self.accum_distances,
                    self.scale)
            dist_rel_err = {'rel_trans': e_trans,
                            'rel_trans_stats': res_writer.compute_statistics(e_trans),
                            #'rel_trans_perc': e_trans_perc,
                            #'rel_trans_perc_stats': res_writer.compute_statistics(e_trans_perc),
                            'rel_rot': e_rot,
                            'rel_rot_stats': res_writer.compute_statistics(e_rot),
                            #'rel_yaw': e_yaw,
                            #'rel_yaw_stats': res_writer.compute_statistics(e_yaw),
                            #'rel_gravity': e_gravity,
                            #'rel_gravity_stats':  res_writer.compute_statistics(e_gravity)
                            }
            self.rel_errors[subtraj_len] = dist_rel_err


    def compute_relative_errors(self, subtraj_lengths=[]):
        if subtraj_lengths:
            for l in subtraj_lengths:
                self.compute_relative_error_at_subtraj_len(l)
        else:
            pcts = [0.1, 0.2, 0.3, 0.4, 0.5]
            print("Computing preset subtrajectory lengths for relative errors...")
            print("Use percentage {0} of trajectory length.".format(pcts))
            subtraj_lengths = [np.floor(pct * self.traj_length)
                               for pct in pcts]

            print("...done. Computed preset subtrajecory lengths:"
                  " {0}".format(subtraj_lengths))
            for l in subtraj_lengths:
                self.compute_relative_error_at_subtraj_len(l)


    def compute_accumulate_errors(self):
        self.accum_distances_est = traj_utils.get_distance_from_start(self.p_es)
        assert len(self.accum_distances_est) == len(self.accum_distances), "estimation and groundtruth should have the same length"
        accum_distance_error = np.abs(self.accum_distances - self.accum_distances_est)
        accum_distances_error_ratio = accum_distance_error[-1] / self.traj_length
        self.accum_errors['accum_distance_error'] = accum_distance_error
        self.accum_errors['accum_distance'] = self.accum_distances
        self.accum_errors['accum_distance_error_ratio'] = accum_distances_error_ratio

