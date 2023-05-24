#!/usr/bin/env python3
# Copyright 2022 MBition GmbH
# SPDX-License-Identifier: MIT
# Taken from https://github.com/mercedes-benz/selfsupervised_flow/blob/master/unsup_flow/datasets/kitti/create.py

import argparse
import os

import numpy as np
import pykitti
from tqdm import tqdm


def convert_structure():

    raw_base_dir = f'{os.path.expanduser("~")}/data/rawkitti/'
    target_dir = f'{os.path.expanduser("~")}/data/rawkitti/prepared/'

    dates = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]

    dimension_dict = {
        "pcl_t0": [-1, 4],
        "pcl_t1": [-1, 4],
        "odom_t0_t1": [4, 4],
    }
    meta = {"framerate__Hz": 10.0}
    skipped_sequences = 0
    success = 0
    for date in tqdm(dates):
        os.makedirs(target_dir + date, exist_ok=True)
        drives_strs = [str(i).zfill(4) for i in range(1000)]

        for drive_str in tqdm(drives_strs):
            print(date, drive_str)
            try:
                kitti = pykitti.raw(raw_base_dir, date, drive_str)
            except FileNotFoundError:
                skipped_sequences += 1
                print("Skipping {0} {1}".format(date, drive_str))
                continue

            os.makedirs(target_dir + date + '/' + drive_str, exist_ok=True)

            for folder in ['lidar', 'pose', 'pairs']:
                os.makedirs(target_dir + date + '/' + drive_str + '/' + folder, exist_ok=True)

            for idx in range(0, len(kitti.velo_files) - 1, 1):
                pcl_t0 = pykitti.utils.load_velo_scan(kitti.velo_files[idx])
                pcl_t1 = pykitti.utils.load_velo_scan(kitti.velo_files[idx + 1])

                w_T_imu_t0 = kitti.oxts[idx].T_w_imu
                w_T_imu_t1 = kitti.oxts[idx + 1].T_w_imu
                imu_T_velo = np.linalg.inv(kitti.calib.T_velo_imu)

                w_T_velo_t0 = np.matmul(w_T_imu_t0, imu_T_velo)
                w_T_velo_t1 = np.matmul(w_T_imu_t1, imu_T_velo)


                odom_t0_t1 = np.matmul(np.linalg.inv(w_T_velo_t0), w_T_velo_t1)
                sample_name = "{0}_{1}_{2}".format(date, drive_str, str(idx).zfill(10))
                data_dict = {
                    "pcl_t0": pcl_t0.astype(np.float32),
                    "pcl_t1": pcl_t1.astype(np.float32),
                    "odom_t0_t1": odom_t0_t1.astype(np.float64),
                    "name": sample_name,
                    "global_pose": w_T_velo_t0,
                }
                # Our protocol
                np.save(target_dir + date + '/' + drive_str + f'/lidar/{idx:06d}.npy', pcl_t0)
                np.save(target_dir + date + '/' + drive_str + f'/pose/{idx:06d}.npy', w_T_velo_t0)
                # SLIM orig
                np.savez(target_dir + date + '/' + drive_str + f'/pairs/{idx:06d}.npz', **data_dict)

                success += 1
    print("Skipped: {0} Success: {1}".format(skipped_sequences, success))


if __name__ == "__main__":
    convert_structure()
