# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function


import warnings
# 모든 경고 메시지 무시 설정
warnings.filterwarnings("ignore")
warnings.filterwarnings("always", category=UserWarning, message=".*GPU.*0.*")


import copy
import random

import numpy as np
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import KITTIOdomDataset
import torch.distributed as dist
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets as datasets
import networks
from IPython import embed

import opticalflow
import cv2
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

torch.autograd.set_detect_anomaly(True)

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed) # torch doc says that torch.manual_seed also work for CUDA
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class Trainer:
    def __init__(self, options):
        self.criterion = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.use_posenet = self.opt.use_posenet
        self.depth_input = self.opt.depth_input
        self.pose_input = self.opt.pose_input
        
        ### multi process setting ###       
        dist.init_process_group(backend='nccl')
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.opt.batch_size = self.opt.batch_size // torch.cuda.device_count()
        torch.cuda.set_device(self.local_rank)
        init_seeds(1+self.local_rank, False)
        
        ### make a savepath ### 
        if dist.get_rank() == 0:
            save_code("./trainer.py", self.log_path)
            save_code("./opticalflow.py", self.log_path)
            save_code("./networks/encoder.py", self.log_path)
            save_code("./networks/depth.py", self.log_path)
            save_code("./networks/resnet_encoder.py", self.log_path)
            save_code("./networks/vit.py", self.log_path)
            save_code("./train.sh", self.log_path)
            if self.opt.use_posenet:
                save_code("./networks/pose.py", self.log_path)
        
        ### model ###
        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cuda")
        self.models.update(self.create_models())
        
        for model_name, model in self.models.items():
            model = model.to(self.device)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.models[model_name] = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
            self.parameters_to_train += list(self.models[model_name].parameters())
        
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate, betas=(self.opt.beta_1, self.opt.beta_2))
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer, milestones=self.opt.milestones, gamma=0.5)
        
        if self.opt.load_weights_folder is not None:
            print("load weight")
            self.load_model()
            
        self.Flow = opticalflow.OpticalFlow(self.opt)
        
        if dist.get_rank() == 0:
            print("Training model named:\n  ", self.opt.model_name)
            print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
            print("Training is using:\n  ", self.device)
            print("Input data is:\n  ", self.depth_input, self.pose_input)


        ### data path ###
        datasets_dict = {"kitti": datasets.KITTIRAWDataset, "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        
        if self.opt.dataset == 'kitti':
            fpath = os.path.join(os.path.dirname(__file__), "./splits", self.opt.split, "{}_files.txt")
            train_filenames = readlines(fpath.format("train"))
            val_filenames = readlines(fpath.format("val"))
            img_ext = '.png' if self.opt.png else '.jpg'

        
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // (self.opt.batch_size * torch.cuda.device_count()) * self.opt.num_epochs
        
        ### dataloader ###
        def worker_init(worker_id):
            worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.novel_frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, sampler=self.train_sampler, pin_memory=True, drop_last=True, worker_init_fn=worker_init, collate_fn=rmnone_collate)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.novel_frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, sampler=self.val_sampler, pin_memory=True, drop_last=True)


        ### self-supervised setting ###
        if self.opt.use_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
            
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
        

        ### record setting ###
        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        if dist.get_rank() == 0:
            self.writers = {}
            for mode in ["train", "val", "test"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
            print("Using split:\n  ", self.opt.split)
            print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))
            self.save_opts()
            
            self.log_file = open(os.path.join(self.log_path, "logs.log"),'w')
            
        self.best_absrel = 10.

        
    def create_models(self):
        models = {}

        if self.depth_input == 'fusion':
            self.models["encoder"] = networks.ResnetEncoder(num_layers=18, num_ch_enc=np.array([64, 64, 128, 256, 512]), pretrained=True, num_input_channels=6)
        else:
            self.models["encoder"] = networks.ResnetEncoder(num_layers=18, num_ch_enc= np.array([64, 64, 128, 256, 512]), pretrained=True, num_input_channels=3)
            
        self.models["depth"] = networks.DepthDecoder(num_ch_enc=np.array([64, 64, 128, 256, 512]),
                                                     num_ch_dec=np.array([16, 32, 64, 128, 256]),
                                                     num_layers=len(np.array([16, 32, 64, 128, 256])),
                                                     scales=self.opt.scales)

        if self.use_posenet:
            if self.pose_input == 'fusion':
                self.models["pose_encoder"] = networks.VIT(input_size=(self.opt.height,self.opt.width), num_input_channels=7)
                self.models["pose"] = networks.PoseDecoder(num_ch_enc=np.array([768]), num_input_features=1, num_frames_to_predict_for=2)
                #self.models["pose_encoder"] = networks.ResnetEncoder(num_layers=18, pretrained=True, num_input_channels=8)
            elif self.pose_input == 'rgb':
                self.models["pose_encoder"] = networks.VIT(input_size=(self.opt.height,self.opt.width), num_input_channels=3)
                self.models["pose"] = networks.PoseDecoder(num_ch_enc=np.array([768]), num_input_features=1, num_frames_to_predict_for=2)
                #self.models["pose_encoder"] = networks.ResnetEncoder(num_layers=18, num_ch_enc= np.array([64, 64, 128, 256, 512]), pretrained=True, num_input_channels=6)
            elif self.pose_input == 'flow':
                self.models["pose_encoder"] = networks.VIT(input_size=(self.opt.height,self.opt.width), num_input_channels=3)
                self.models["pose"] = networks.PoseDecoder(num_ch_enc=np.array([768]), num_input_features=1, num_frames_to_predict_for=2)
                #self.models["pose_encoder"] = networks.ResnetEncoder(num_layers=18, pretrained=True, num_input_channels=2)

        return models
        

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()
            
    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():

            m.eval()
            
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        for self.epoch in range(self.opt.start_epoch):
            self.model_lr_scheduler.step()
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self.Flow.num = 0
            self.Flow.mean = 0
            self.Flow.var = 0
            
            self.run_epoch()
            self.evaluate_depth()
            self.evaluate_pose()
            if dist.get_rank() == 0:
                self.save_model("last_models")
                with open("example.txt", 'a') as file:
                    file.write("\n   Flow mean: " + str(self.Flow.mean/self.Flow.num) + "   Flow var: " + str(self.Flow.var/self.Flow.num) + "   Flow max: " + str(self.Flow.max) + "\n")
                    file.write("\n")
 
                
    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        
        self.train_sampler.set_epoch(self.epoch)
        self.set_train()
        self.models["encoder"].eval()
        self.models["depth"].eval()
        for batch_idx, inputs in enumerate(self.train_loader):
            if inputs is None:
                self.model_optimizer.zero_grad()
                self.model_optimizer.step()
                self.step += 1
                continue
            before_op_time = time.time()
            
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            early_phase = batch_idx % 10 == 0 and self.step < self.opt.log_frequency
            late_phase = self.step % self.opt.log_frequency == 0
            if early_phase or late_phase:
                if dist.get_rank() == 0:
                    self.log_time(batch_idx, duration, losses)
                    losses.update(self.compute_depth_losses(inputs, outputs))
                    self.log("train", losses)

            self.step += 1
            if batch_idx % self.opt.log_img_frequency == 0 and dist.get_rank() == 0:
                self.log_img("train", inputs, outputs, batch_idx)
                
        self.val()
        self.model_lr_scheduler.step()


    def process_batch(self, inputs):
        # Pass a minibatch through the network and generate images and losses
        B,_,H,W = inputs["color_aug", 0, 0].shape
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
            
        inputs["flow", 1, 0] = self.Flow([inputs["color_aug", 0, 0], inputs["color_aug", 1, 0]])[0].detach()
        inputs["flow", -1, 0] = self.Flow([inputs["color_aug", 0, 0], inputs["color_aug", -1, 0]])[0].detach()

        outputs = {}
        disp_sum = torch.zeros((B,1,H,W)).cuda()
            
        if self.depth_input == "flow":
            features = self.models['encoder'](inputs["flow", self.image_for_flow, 0])
            outputs = self.models['depth'](features)
            
        elif self.depth_input == "fusion":
            features = self.models['encoder'](torch.cat([preprocess_rgb(inputs["color_aug", 0, 0]), preprocess_flow(inputs["flow", self.image_for_flow, 0])], dim=1))
            outputs = self.models['depth'](features)
            
        elif self.depth_input == "rgb":
            features = self.models['encoder'](preprocess_rgb(inputs["color_aug", 0, 0]))
            outputs.update(self.models['depth'](features))
        
        for scale in self.opt.scales:
            disp = outputs[("disp", 0, scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            disp_sum += disp
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            
            outputs[("depth", 0, scale)] = depth
            
        
        disp_sum /= len(self.opt.scales)
        outputs.update(self.predict_poses(inputs, disp_sum))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        
        
        return outputs, losses
        

    def predict_poses(self, inputs, depth):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        
        if self.pose_input == "rgb":
            pose_feats = {f_i: preprocess_rgb(inputs["color_aug", f_i, 0]) for f_i in self.opt.novel_frame_ids}

            for f_i in self.opt.novel_frame_ids[1:]:
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = self.models["pose_encoder"](torch.cat(pose_inputs, 1))
                axisangle, translation = self.models["pose"](pose_inputs[-1])
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
    
                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                
                
        elif self.pose_input == 'flow':
            for f_i in self.opt.novel_frame_ids[1:]:
                pose_inputs, attn_map = self.models["pose_encoder"](preprocess_flow(inputs["flow", f_i, 0]))
                pose_inputs, attn_map = self.models["pose_encoder"](pose_inputs)
                attn_map_min, attn_map_max, attn_map_mean = attn_map
                
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("attn", 0, f_i)] = attn_map
                
                outputs[("attn_max", 0, f_i)] = attn_map_max
                outputs[("attn_min", 0, f_i)] = attn_map_min
                outputs[("attn_mean", 0, f_i)] = attn_map_mean

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
          
                
        elif self.pose_input == 'fusion':
            for f_i in self.opt.novel_frame_ids[1:]:
                pose_inputs = torch.cat([preprocess_rgb(inputs["color_aug", 0, 0]), preprocess_disp(depth), preprocess_flow(inputs["flow", f_i, 0])], dim=1)
                pose_inputs, attn_map = self.models["pose_encoder"](pose_inputs)

                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                
                attn_map, masked_attn_map = attn_map
                outputs[("attn_min", 0, f_i)] = attn_map
                outputs[("masked_attn_min", 0, f_i)] = masked_attn_map

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                
        return outputs


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            source_scale = 0
            depth = outputs["depth", 0, scale]

            for i, frame_id in enumerate(self.opt.novel_frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)] 

                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
                
                outputs[("sample", frame_id, scale)] = pix_coords
                
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.use_ssim:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        else:
            reprojection_loss = l1_loss
            
        return reprojection_loss
    

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0
        
        for frame_id in self.opt.novel_frame_ids[1:]:
            outputs[("mask", frame_id, 0)] = create_mask(outputs[("attn_min", 0, frame_id)], (self.opt.height, self.opt.width))   # create mask generated from attention map
        

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            ##### For minimum reprojection loss #####
            for frame_id in self.opt.novel_frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            
            
            identity_reprojection_losses = []
            for frame_id in self.opt.novel_frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))           
            
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_loss = identity_reprojection_losses
            reprojection_loss = reprojection_losses

            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 0.00001
            
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            
            to_optimise, idxs = torch.min(combined, dim=1)
            outputs["identity_selection/{}".format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()
            loss += to_optimise.mean()
                
            if scale == 0:
                outputs[("lossmap")] = to_optimise.clone().detach()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            

        total_loss /= len(self.opt.scales)
        losses["loss"] = total_loss
        return losses
    
    
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        num = 0
        metrics = {}

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                 

                outputs, losses = self.process_batch(inputs)
                if batch_idx % self.opt.log_frequency == 0 and self.local_rank == 0:
                    self.log("val", losses)

                losses = self.compute_depth_losses(inputs, outputs)
                
                B = inputs[("color", 0, 0)].shape[0]
                num += B
                for k,v in losses.items():
                    if k in metrics:
                        metrics[k] += v * B
                    else:
                        metrics[k] = v * B
                        
                if batch_idx % (self.opt.log_img_frequency) == 0 and self.local_rank == 0:
                        self.log_img("val", inputs, outputs, batch_idx)
            
                
            # since the eval batch size is not the same
            # we need to sum them then mean   
            num = torch.ones(1).cuda() * num
            dist.all_reduce(num, op=dist.ReduceOp.SUM)
            for k,v in metrics.items():
                dist.all_reduce(metrics[k], op=dist.ReduceOp.SUM)
                metrics[k] = metrics[k] / num
            if metrics["de/abs_rel"] < self.best_absrel:
                self.best_absrel = metrics["de/abs_rel"]
                if self.local_rank == 0:
                    self.save_model("best_models")
                    
                    
            if self.local_rank == 0:
                self.log("val", metrics)
                print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                print(("&{: 8.4f}  " * 7).format(*[metrics[k].cpu().data[0] for k in self.depth_metric_names]) + "\\\\")
                #write to log file
                print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"), file=self.log_file)
                print(("&{: 8.4f}  " * 7).format(*[metrics[k].cpu().data[0] for k in self.depth_metric_names]) + "\\\\", file=self.log_file)
                
            del inputs, outputs, losses
        self.set_train()
    
    

    def compute_depth_losses(self, inputs, outputs):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        losses = {}
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            #losses[metric] = np.array(depth_errors[i].cpu())
            losses[metric] = depth_errors[i]
            
        return losses
    
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = self.log_path
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, folder_name):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, folder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.module.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].module.state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].module.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
            
            
    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size * torch.cuda.device_count() / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss"].cpu().data, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        
        
    def log_img(self, mode, inputs, outputs, val_idx):
        """Write an event to the tensorboard events file
        """
        
        writer = self.writers[mode]
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for frame_id in self.opt.novel_frame_ids:
                writer.add_image("color_{}/{}".format(frame_id, self.epoch), inputs[("color", frame_id, 0)][j].data, val_idx+j)
                if frame_id != 0:
                    writer.add_image("color_pred_{}/{}".format(frame_id, self.epoch), outputs[("color", frame_id, 0)][j].data, val_idx+j)
                    writer.add_image("flow_map_{}/{}".format(frame_id, self.epoch), normalize_image(inputs["flow", frame_id, 0][j] * 255), val_idx+j)
                    writer.add_image("mask_{}/{}".format(frame_id, self.epoch), outputs[("mask", frame_id, 0)][j] * inputs[("color", 0, 0)][j], val_idx+j)
                    writer.add_image("attn_map_min_{}/{}".format(frame_id, self.epoch),
                             self.show_attn_on_image(inputs["color", 0, 0][j], outputs[("attn_min", 0, frame_id)][j]).transpose(2,0,1), val_idx+j)
                    writer.add_image("attn_map_inverse_min_{}/{}".format(frame_id, self.epoch),
                             self.show_attn_on_image(inputs["color", 0, 0][j], outputs[("attn_min", 0, frame_id)][j], inverse=True).transpose(2,0,1), val_idx+j)
                    writer.add_image("masked_attn_map_min_{}/{}".format(frame_id, self.epoch),
                             self.show_attn_on_image(inputs["color", 0, 0][j], outputs[("masked_attn_min", 0, frame_id)][j]).transpose(2,0,1), val_idx+j)
                    writer.add_image("masked_attn_map_inverse_min_{}/{}".format(frame_id, self.epoch),
                             self.show_attn_on_image(inputs["color", 0, 0][j], outputs[("masked_attn_min", 0, frame_id)][j], inverse=True).transpose(2,0,1), val_idx+j)
                    writer.add_image("pure_attn_map_{}/{}".format(frame_id, self.epoch),normalize_image(outputs["attn_min", 0, frame_id][j].unsqueeze(0)), val_idx+j)

                    
            writer.add_image("mask_or/{}".format(self.epoch), (((outputs[("mask", -1, 0)][j] + outputs[("mask", 1, 0)][j])) / 2) * inputs[("color", 0, 0)][j], val_idx+j)
            writer.add_image("mask_and/{}".format(self.epoch), (outputs[("mask", -1, 0)][j] * outputs[("mask", 1, 0)][j]) * inputs[("color", 0, 0)][j], val_idx+j)         
            writer.add_image("disp/{}".format(self.epoch),self.make_depthmap(disp = outputs["disp", 0, 0][j].unsqueeze(0).detach()), val_idx+j)
            writer.add_image("depth/{}".format(self.epoch),normalize_image(outputs["depth", 0, 0][j]), val_idx+j)
            writer.add_image("automask/{}".format(self.epoch), outputs["identity_selection/0"][j][None, ...], val_idx+j)
            writer.add_image("lossmap/{}".format(self.epoch), self.show_attn_on_image(inputs["color", 0, 0][j], outputs[("lossmap")][j]).transpose(2,0,1), val_idx+j)
            

    def log(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar(l, v, self.step)
            
            
    def evaluate_depth(self):
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        
        test_filenames = readlines(os.path.join(os.path.join(os.path.dirname(__file__), "splits"), self.opt.eval_split, "test_files.txt"))
        test_dataset = self.dataset(self.opt.data_path, test_filenames, self.opt.height, self.opt.width, self.opt.novel_frame_ids, 4, is_train=False, img_ext='.png')
        test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)  #Drop last 꺼야함

        if self.opt.ext_disp_to_eval is None:
            
            self.models["encoder"].eval()
            self.models["depth"].eval()
            pred_disps = []
            with torch.no_grad():
                for inputs in test_loader:     
                    
                    for key, ipt in inputs.items():
                        inputs[key] = ipt.to(self.device)          
                        
                    inputs["flow", 1, 0] = self.Flow([inputs["color_aug", 0, 0], inputs["color_aug", 1, 0]])[0].detach()  

                    if self.depth_input == "flow":
                        features = self.models['encoder'](inputs["flow", self.image_for_flow, 0])
                        outputs = self.models['depth'](features)
                        
                    elif self.depth_input == "fusion":
                        features = self.models['encoder'](torch.cat([preprocess_rgb(inputs["color_aug", 0, 0]), preprocess_flow(inputs["flow", 1, 0])], dim=1))
                        outputs = self.models['depth'](features)
                    
                    elif self.depth_input == "rgb":
                        features = self.models['encoder'](preprocess_rgb(inputs["color_aug", 0, 0]))
                        outputs = self.models['depth'](features)


                    pred_disp, _ = disp_to_depth(outputs[("disp", 0, 0)], self.opt.min_depth, self.opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()
                    pred_disps.append(pred_disp)

    
            pred_disps = np.concatenate(pred_disps)
        
        gt_path = os.path.join(os.path.join(os.path.dirname(__file__), "splits"), self.opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        errors = []
        ratios = []

        for i in range(pred_disps.shape[0]):
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            if self.opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            else:
                mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= self.opt.pred_depth_scale_factor
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            errors.append(self.compute_errors(gt_depth, pred_depth))

        mean_errors = np.array(errors).mean(0)
        
        if dist.get_rank() == 0:
            if not self.opt.disable_median_scaling:
                ratios = np.array(ratios)
                med = np.median(ratios)
                print("   Mono evaluation - using median scaling")
                print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
                
            with open("example.txt", 'a') as file:
                file.write("\n" + "   Epoch: " + str(self.epoch))
                file.write("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                file.write("\n " + ("{: 8.3f}   " * 7).format(*mean_errors.tolist()))
                file.write("\n")
            print("\n-> Done!")
            
            losses = {}
            mean_errors = mean_errors.tolist()
            for i, metric in enumerate(self.depth_metric_names):
                losses[metric] = mean_errors[i]
                
            self.log("test", losses)
        
    
    def make_depthmap(self, disp):
        disp_resized = torch.nn.functional.interpolate(disp, (384, 1280), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8).transpose(2,0,1)
        
        return colormapped_im
    

    def compute_errors(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    
    
    def show_attn_on_image(self, img, attn, inverse = False):
        attn_copy = attn.clone().cpu().detach()
        min_value = torch.min(attn_copy)
        max_value = torch.max(attn_copy)
        attn_copy = (attn_copy - min_value) / (max_value - min_value)
        if inverse:
            attn_copy = torch.clip(attn_copy, min=0.0, max=torch.mean(attn_copy) - torch.std(attn_copy))
            min_value = torch.min(attn_copy)
            max_value = torch.max(attn_copy)
            attn_copy = (attn_copy - min_value) / (max_value - min_value)
            attn_copy = 1 - attn_copy
        

        attn_copy = attn_copy.numpy()
        attn_copy = cv2.resize(attn_copy, (self.opt.width, self.opt.height))
        attn_copy = cv2.cvtColor(attn_copy, cv2.COLOR_RGB2BGR)

        img_copy = img.clone().cpu().detach()
        img_copy = img_copy.permute(1,2,0).numpy()
        img_copy = cv2.resize(img_copy, (self.opt.width, self.opt.height))
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        img_copy = np.float32(img_copy)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_copy), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img_copy)
        cam = cam / np.max(cam)

        return np.uint8(255 * cam)


    def evaluate_pose(self):
        """Evaluate odometry on the KITTI dataset
        """
        
        pose_split = "odom_9"   # "odom_10"

        sequence_id = int(pose_split.split("_")[1])

        filenames = readlines(os.path.join(os.path.dirname(__file__), "splits", "odom", "test_files_{:02d}.txt".format(sequence_id)))

        dataset = KITTIOdomDataset(self.opt.pose_path, filenames, self.opt.height, self.opt.width, [0, 1], 4, is_train=False, img_ext=".png")
        dataloader = DataLoader(dataset, self.opt.batch_size, shuffle=False, num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)

    
        self.models["pose_encoder"].eval()
        self.models["pose"].eval()
        pred_poses = []
          
        with torch.no_grad():
            for inputs in dataloader:
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda()
                
                if self.pose_input == "rgb":
                    all_color_aug = torch.cat([preprocess_rgb(inputs[("color_aug", 0, 0)]), preprocess_rgb(inputs[("color_aug", 1, 0)])], 1)
                    features = self.models["pose_encoder"](all_color_aug)
                    axisangle, translation = self.models["pose"](features[-1])
                
                elif self.pose_input == 'fusion':
                    inputs["flow", 1, 0] = self.Flow([inputs["color_aug", 0, 0], inputs["color_aug", 1, 0]])[0].detach()
                    features = self.models['encoder'](preprocess_rgb(inputs["color_aug", 0, 0]))
                    outputs = self.models['depth'](features)
                    _, depth = disp_to_depth(outputs[("disp", 0, 0)], self.opt.min_depth, self.opt.max_depth)
                    pose_inputs = torch.cat([preprocess_rgb(inputs["color_aug", 0, 0]), preprocess_depth(depth), preprocess_flow(inputs["flow", 1, 0])], dim=1)
                    pose_inputs, attn_map = self.models["pose_encoder"](pose_inputs)
                    axisangle, translation = self.models["pose"](pose_inputs)
                    
                pred_poses.append(transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

        pred_poses = np.concatenate(pred_poses)
        gt_poses_path = os.path.join(self.opt.pose_path, "poses", "{:02d}.txt".format(sequence_id))
        gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
        gt_global_poses = np.concatenate((gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
        gt_global_poses[:, 3, 3] = 1
        gt_xyzs = gt_global_poses[:, :3, 3]

        gt_local_poses = []
        for i in range(1, len(gt_global_poses)):
            gt_local_poses.append(np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

        ates = []
        num_frames = gt_xyzs.shape[0]
        track_length = 5
        for i in range(0, num_frames - 1):
            local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
            gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

            ates.append(compute_ate(gt_local_xyzs, local_xyzs))
        

        if dist.get_rank() == 0:
            with open("example.txt", 'a') as file:
                file.write("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))
                file.write("\n")
            print("\n-> Done!")

                  
        
        
def preprocess_rgb(input_image):
    return (input_image - 0.45) / 0.225

def preprocess_flow(input_image):
    return (input_image - 0.9826) / 0.0283

def preprocess_depth(input_image, max=100, min=0.1):
    process_depth = (input_image - min) / (max - min)
    return (process_depth - 0.45) / 0.225

def preprocess_disp(input_image):
    process_disp = (input_image - input_image.mean()) / input_image.std()
    return process_disp

def create_mask(attn_map, input_shape, scale_factor=1):
    threshold = (torch.mean(attn_map, dim=(1,2)) - scale_factor * torch.std(attn_map, dim=(1,2))).cuda()
    mask = []
    for i in range(attn_map.size(0)):
        thresholded_batch = torch.where(attn_map[i] <= threshold[i], torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())
        mask.append(thresholded_batch)

    mask = torch.stack(mask).unsqueeze(1)
    mask = F.interpolate(mask, input_shape, mode="nearest")
    
    return mask


def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def attn_entropy(attn_input):
    # attn_input: (Batch_size, H, W, H, W)
    # attn_output: (Batch_size, H, W)
    # entropy_output: (Batch_size, H, W)
    B, H, W, _, _ = attn_input.size()
   
    attn_output = torch.zeros(B, H, W).cuda()
    self_attn_output = torch.zeros(B, H, W).cuda()
    
    for b in range(B):
        for h in range(H):
            for w in range(W):
                attn_output[b] += attn_input[b, h, w]
                attn_output[b, h, w] -= attn_input[b, h, w, h, w]
                self_attn_output[b, h, w] += attn_input[b, h, w, h, w]
    
    entropy_output = -((attn_input + 1e-7) * torch.log2(attn_input + 1e-7)).sum(dim=(3, 4)).cuda()

    return attn_output, entropy_output, self_attn_output