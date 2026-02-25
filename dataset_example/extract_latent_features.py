import mediapy
import os
import torch
import numpy as np
import json
import pandas as pd
from torch.utils.data import Dataset
from accelerate import Accelerator
import sys
from PIL import Image
import torch.nn.functional as F
import cv2
import mediapy as media
import shutil

#Segmentation
base_dir = os.path.dirname(os.path.abspath(__file__))
GSAM_ROOT = os.path.join(base_dir, "Grounded-SAM-2")
sys.path.append(GSAM_ROOT)

sys.path.append(os.path.join(GSAM_ROOT, "grounding_dino"))
sys.path.append(os.path.join(GSAM_ROOT, "sam2"))

#Optical_flow
sys.path.append(os.path.join(base_dir, "bridge_training_code"))

#seg2
from sam2.build_sam import build_sam2_video_predictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

#optical flow
from gmflow.gmflow.gmflow import GMFlow

def load_image_from_rgb(image_rgb_uint8: np.ndarray):
    """
    image_rgb_uint8: (H,W,3) RGB uint8
    returns: (image_pil, image_tensor) where image_tensor is (3,h,w) normalized
    """
    image_pil = Image.fromarray(image_rgb_uint8).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image

class ExtractFeatureDataset(Dataset): 
    def __init__(self, old_path, new_path, device, size=(192, 320), rgb_skip=3, extra_feature=None):
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.device = device
        self.extra_feature=extra_feature
        self.extra_feature = extra_feature

        # ====== GSAM 2======
        if self.extra_feature == "seg2":
            self.sam2_checkpoint = "/scr/hyeonhoo/checkpoints/gsam2/checkpoints/sam2.1_hiera_large.pt"
            self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device=self.device, apply_postprocessing=False,)
            
            model_id = "IDEA-Research/grounding-dino-base"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
            
            self.box_threshold = 0.3
            self.text_threshold = 0.35
        
        #Optical_flow
        elif self.extra_feature == "optical_flow":
            self.inference_size = [480, 480]
            self.horizon = 1 # subsampled frame gap

            # GMFlow
            self.flow_model = GMFlow(
                feature_channels=128,
                num_scales=1,
                upsample_factor=8,
                num_head=1,
                attention_type='swin',
                ffn_dim_expansion=4,
                num_transformer_layers=6,
            ).to(device)

            ckpt_path = '/scr/hyeonhoo/checkpoints/optical_flow/pretrained/gmflow_sintel-0c07dcb3.pth'
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
                self.flow_model.load_state_dict(weights, strict=False)
                self.flow_model.eval()
                print(f"Loaded GMFlow checkpoint from {ckpt_path}")
            else:
                print(f"Warning: GMFlow checkpoint not found at {ckpt_path}")

        def load_json_file(file_path):
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))  # 使用 json.loads() 解析单行
            return data

        self.data = load_json_file(f'{old_path}/meta/episodes.jsonl')

    def _save_overlay_video(self, raw_frames, feature_list, save_path, fps=5, boxes_xyxy=None):
        """
        raw_frames: (T,H,W,3) uint8 RGB
        feature_list: list of (1,h,w) torch tensor (cpu)
        boxes_xyxy: None or (N,4) np.ndarray in XYXY pixel coords for frame 0 DINO)
        """
        vis = []
        for t in range(len(feature_list)):
            frame = raw_frames[t].copy()  # RGB uint8

            # ---- mask overlay ----
            mask = feature_list[t][0].numpy()
            mask = (mask > 0.5).astype(np.uint8)

            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            frame[..., 0] = np.maximum(frame[..., 0], mask * 255)  # R channel

            # ---- bbox overlay (optional) ----
            # if boxes_xyxy is not None and len(boxes_xyxy) > 0:
            #     for box in boxes_xyxy:
            #         x1, y1, x2, y2 = [int(v) for v in box]
            #         x1 = max(0, min(x1, frame.shape[1]-1))
            #         x2 = max(0, min(x2, frame.shape[1]-1))
            #         y1 = max(0, min(y1, frame.shape[0]-1))
            #         y2 = max(0, min(y2, frame.shape[0]-1))
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            vis.append(frame)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media.write_video(save_path, vis, fps=fps)


    #optical flow
    @torch.no_grad()
    def compute_flow_batch(self, img1, img2, device, target_size=None):
        """
        img1, img2: (B, 3, H, W) Normalized [-1, 1] tensor
        target_size: (H_out, W_out) tuple. If None, revert to original image size.
        Returns: (B, 2, H_out, W_out) Optical Flow
        """
        #[-1, 1] -> [0, 255]
        img1 = (img1 * 0.5 + 0.5) * 255.0
        img2 = (img2 * 0.5 + 0.5) * 255.0
        
        if target_size is not None:
            out_size = target_size
        else:
            out_size = img1.shape[-2:] # (H, W)


        img1_in = F.interpolate(img1, size=self.inference_size, mode='bilinear', align_corners=True)
        img2_in = F.interpolate(img2, size=self.inference_size, mode='bilinear', align_corners=True)
        

        results_dict = self.flow_model(
            img1_in, img2_in,
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
            pred_bidir_flow=False,
        )
        

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H_inf, W_inf]
        flow_pr = F.interpolate(flow_pr, size=out_size, mode='bilinear', align_corners=True)
        
        flow_pr[:, 0] = flow_pr[:, 0] * out_size[-1] / self.inference_size[-1] # W
        flow_pr[:, 1] = flow_pr[:, 1] * out_size[-2] / self.inference_size[-2] # H
        
        # ===== Visualization=====
        Visualization = False
        if Visualization:
            if not hasattr(self, "_debug_flow_frames"):
                self._debug_flow_frames = []
                self._debug_flow_saved = False

            MAX_DEBUG_FRAMES = 240  
            DEBUG_FPS = 5 


            if len(self._debug_flow_frames) < MAX_DEBUG_FRAMES:
    
                frame_vis = img1[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                
                flow_np = flow_pr[0].permute(1, 2, 0).cpu().numpy() # (H, W, 2)
                h, w = flow_np.shape[:2]
                
                hsv = np.zeros((h, w, 3), dtype=np.uint8)
                hsv[..., 1] = 255
                mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                if frame_vis.shape[:2] != flow_vis.shape[:2]:
                    frame_vis = cv2.resize(frame_vis, (w, h))

                combined = np.concatenate([frame_vis, flow_vis], axis=1)
                self._debug_flow_frames.append(combined)


            if (len(self._debug_flow_frames) >= MAX_DEBUG_FRAMES) and (not self._debug_flow_saved):
                self._debug_flow_saved = True

                save_dir = os.path.join("dataset_example/outputs/", "debug_vis")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "optical_flow_debug.mp4")

                media.write_video(save_path, self._debug_flow_frames, fps=DEBUG_FPS)
                print(f"[DEBUG] Saved Optical Flow video to {save_path} (frames={len(self._debug_flow_frames)})", flush=True)
        # ================================================================

        return flow_pr # (B, 2, H, W)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_data = self.data[idx]
        instruction = traj_data['tasks'][0]
        traj_id = traj_data['episode_index']
        chunk_id = int(traj_id/1000)

        data_type = 'val' if traj_id%100 == 99 else 'train'
        # if os.path.exists(f"{self.new_path}/videos/{data_type}/{traj_id}") and os.path.exists(f"{self.new_path}/latent_videos/{data_type}/{traj_id}"):
        #     # print(f"Skipping trajectory {traj_id}, already processed.")
        #     return 0

        file_path = f'{self.old_path}/data/chunk-{chunk_id:03d}/episode_{traj_id:06d}.parquet'
        df = pd.read_parquet(file_path)
        length = len(df['observation.state.cartesian_position'])

        obs_car = []
        obs_joint =[]
        obs_gripper = []
        action_car = []
        action_joint = []
        action_gripper = []
        action_joint_vel = []

        for i in range(length):
            obs_car.append(df['observation.state.cartesian_position'][i].tolist())
            obs_joint.append(df['observation.state.joint_position'][i].tolist())
            obs_gripper.append(df['observation.state.gripper_position'][i].tolist())
            action_car.append(df['action.cartesian_position'][i].tolist())
            action_joint.append(df['action.joint_position'][i].tolist())
            action_gripper.append(df['action.gripper_position'][i].tolist())
            action_joint_vel.append(df['action.joint_velocity'][i].tolist())
        success = df['is_episode_successful'][0]
        video_paths = [
                    f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.exterior_1_left/episode_{traj_id:06d}.mp4',
                    f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.exterior_2_left/episode_{traj_id:06d}.mp4',
                    f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.wrist_left/episode_{traj_id:06d}.mp4']

        # if f"{save_root}/videos/{data_type}/{traj_id}" exist, skip this trajectory
        try:
            self.process_traj(video_paths, instruction, self.new_path, traj_id=traj_id, data_type=data_type, size=self.size, rgb_skip=self.skip)
        except Exception as e:
            print(f"Error processing trajectory {traj_id}: {e}")
            return 0
    
        return 0


    def process_traj(self, video_paths, instruction, save_root,traj_id=0, data_type='val', size=(192,320), rgb_skip=3):
        for video_id, video_path in enumerate(video_paths):
            # load and resize video and save
            video = mediapy.read_video(video_path)
            
            if self.extra_feature == "seg2":
                raw_frames = np.array(video[::rgb_skip])
                if raw_frames.dtype != np.uint8:
                    raw_frames = (np.clip(raw_frames, 0, 1) * 255).astype(np.uint8)
            else:
                frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0*2-1
                frames = frames[::rgb_skip]  # Skip frames to save memory here!!!

            feature_list = []
            with torch.no_grad():
                if self.extra_feature == "seg2":
                    T = raw_frames.shape[0]

                    # ---------- 1) dump frames to a temp directory ----------
                    tmp_dir = os.path.join("dataset_example/seg2_tmp", "sam2_frames", f"traj_{traj_id:06d}", f"cam_{video_id}")
                    os.makedirs(tmp_dir, exist_ok=True)

                    for i in range(T):
                        bgr = cv2.cvtColor(raw_frames[i], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(tmp_dir, f"{i}.jpg"), bgr)

                    # ---------- 2) init_state(video_path=...) ----------
                    inference_state = self.video_predictor.init_state(video_path=tmp_dir)

                    # ---------- 3) Grounding DINO on first frame ----------
                    first_frame_pil = Image.fromarray(raw_frames[0]).convert("RGB")

                    text_prompt = str(instruction).lower().strip()
                    if not text_prompt.endswith("."):
                        text_prompt += "."

                    inputs = self.processor(images=first_frame_pil, text=text_prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.grounding_model(**inputs)

                    W, H = first_frame_pil.size
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        target_sizes=[(H, W)],
                    )

                    input_boxes = results[0]["boxes"].detach().to("cpu").numpy().astype(np.float32)  # (N,4)

                    # ---------- 4) register boxes ----------
                    if len(input_boxes) > 0:
                        for obj_id, box in enumerate(input_boxes, start=1):
                            self.video_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=obj_id,
                                box=box,
                            )

                    # ---------- 5) propagate & build per-frame combined mask ----------
                    video_segments = {}
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                        if len(out_obj_ids) > 0:
                            m = (out_mask_logits > 0.0)
                            # m: (N,1,H,W) or (N,H,W)
                            if m.ndim == 4:
                                m = m[:, 0]              # (N,H,W)
                            combined = m.any(dim=0)       # (H,W)
                            combined_mask = combined[None].float()  # (1,H,W)
                        else:
                            combined_mask = torch.zeros((1, H, W), device=self.device)

                        if size is not None:
                            combined_mask = F.interpolate(
                                combined_mask.unsqueeze(0),  # (1,1,H,W)
                                size=size,
                                mode="nearest"
                            ).squeeze(0)  # (1,H,W)

                        video_segments[out_frame_idx] = combined_mask.detach().to("cpu")

                    # ---------- 6) ensure all frames exist ----------
                    feature_list = []
                    for i in range(T):
                        if i in video_segments:
                            feature_list.append(video_segments[i])
                        else:
                            z = torch.zeros((1, size[0], size[1])) if size is not None else torch.zeros((1, H, W))
                            feature_list.append(z)

                    self.video_predictor.reset_state(inference_state)

                    #====Visualization=====
                    # save_path = os.path.join(
                    #     "dataset_example", "outputs", "seg2",
                    #     data_type, f"{traj_id:06d}", f"{video_id}.mp4")
                    # self._save_overlay_video(raw_frames, feature_list, save_path, fps=5, boxes_xyxy=input_boxes)

                    #remove image folder
                    shutil.rmtree(tmp_dir, ignore_errors=False)

                elif self.extra_feature == "optical_flow":
                    num_frames = len(frames)
                    frames = frames.to(self.device)

                    for i in range(0, num_frames -1):
                        img1 = frames[i].unsqueeze(0)
                        img2 = frames[i + 1].unsqueeze(0)
                        
                        flows = self.compute_flow_batch(img1, img2, self.device, target_size=size)
                        feature_list.append(flows[0].cpu())
                    
                    if feature_list:
                        feature_list.append(torch.zeros_like(feature_list[-1]))  # (2,H,W)
                elif self.extra_feature == "gipper_detection":
                    pass #TODO
            if self.extra_feature == "seg2":
                traj_base_dir = os.path.join("dataset_example/seg2_tmp", "sam2_frames", f"traj_{traj_id:06d}")
                shutil.rmtree(traj_base_dir, ignore_errors=False)    


            final_features = torch.stack(feature_list, dim=0)
            os.makedirs(f"{save_root}/latent_videos/{self.extra_feature}/{data_type}/{traj_id}", exist_ok=True)
            torch.save(final_features, f"{save_root}/latent_videos/{self.extra_feature}/{data_type}/{traj_id}/{video_id}.pt")

if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--droid_hf_path', type=str, default='/scr/shared/world_model/DROID-1.0.1')
    parser.add_argument('--droid_output_path', type=str, default='/scr/hyeonhoo/outputs/extra_features')
    parser.add_argument('--extra_feature', type=str, default=None)
    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    accelerator = Accelerator()
    dataset = ExtractFeatureDataset(
        old_path=args.droid_hf_path,
        new_path= args.droid_output_path,
        device=accelerator.device,
        rgb_skip=5, #  to downsample 15hz video to 5hz video
        size=(192, 320),
        extra_feature=args.extra_feature,
    )
    tmp_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
        ) 
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    for idx, _ in enumerate(tmp_data_loader):
        if idx == 5 and args.debug:
            break
        if idx % 100 == 0 and accelerator.is_main_process:
            print(f"Precomputed {idx} samples")