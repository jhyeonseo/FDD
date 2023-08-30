# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
import argparse


file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_dataset"))
        self.parser.add_argument("--pose_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_dataset_odom"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="results")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--depth_input",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="rgb")
        self.parser.add_argument("--pose_input",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="rgb")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "eigen_full_left", "odom", "benchmark"],
                                 default="eigen_full_left")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_ssim",
                                 help="if set, use ssim in the loss",
                                 default="store_true")
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--novel_frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1,  1])
        self.parser.add_argument("--no_crop",
                                 action="store_true",
                                 help="if set, do not use resize crop data aug")
        self.parser.add_argument("--use_colmap",
                                 action="store_true",
                                 help="if set, use colmap instead of predicting pose by posenet")
        self.parser.add_argument("--colmap_path",
                                 type=str,
                                 help="path to the colmap data",
                                 default="./kitti_colmap")
        self.parser.add_argument('--use_posenet', 
                                 default="store_true", help="use posenet instead of 5-point alignment algorithm")
        self.parser.add_argument("--flip_right",
                                 action="store_true",
                                 help="use fliped right image to train")
        self.parser.add_argument("--match_aug",
                                 action="store_true",
                                 help="if set, use color augmented data to compute loss")
        self.parser.add_argument("--log_img_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=100)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])                      # default=[0, 1, 2, 3]

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=16)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--beta_1",
                                 type=float,
                                 help="beta1 of Adam",
                                 default=0.5)
        self.parser.add_argument("--beta_2",
                                 type=float,
                                 help="beta2 of Adam",
                                 default=0.999)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--start_epoch",
                                 type=int,
                                 help="number of epochs",
                                 default=0)
        self.parser.add_argument('--milestones', 
                                 default=[30, 40], nargs='*',
                                 help='epochs at which learning rate is divided by 2')
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="epochs at which learning rate times 0.1",
                                 default=20)

        # ABLATION options
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")

        # SYSTEM options
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 #default=["encoder", "depth"])
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=100)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        
        # Raft options
        self.parser.add_argument('--flow_model', default="models/raft-kitti.pth", help="restore checkpoint")
        self.parser.add_argument('--path', help="dataset for evaluation")
        self.parser.add_argument('--small', action='store_true', help='use small model')
        self.parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        self.parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')        


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
