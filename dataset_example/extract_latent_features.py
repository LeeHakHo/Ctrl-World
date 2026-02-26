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
import re

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
from groundingdino.util.inference import load_model
from groundingdino.util.inference import predict
import groundingdino.datasets.transforms as GD_T
from groundingdino.util.box_ops import box_cxcywh_to_xyxy

#optical flow
from gmflow.gmflow.gmflow import GMFlow

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

            #SAM 2
            self.sam2_checkpoint = "/scr/hyeonhoo/checkpoints/gsam2/checkpoints/sam2.1_hiera_large.pt"
            self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device=self.device, apply_postprocessing=False,)
            
            #DINO
            # self.gdino_config = os.path.join(GSAM_ROOT, "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py")
            # self.gdino_checkpoint = "/scr/hyeonhoo/checkpoints/gsam2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
            # self.grounding_model = load_model(self.gdino_config, self.gdino_checkpoint, device=self.device)
            
            #self.box_threshold = 0.3
            #self.text_threshold = 0.35

            # Florence-2
            from transformers import AutoProcessor, AutoModelForCausalLM
            self.florence_model_id = "microsoft/Florence-2-large"
            self.florence_processor = AutoProcessor.from_pretrained(self.florence_model_id, trust_remote_code=True)
            self.florence_model = AutoModelForCausalLM.from_pretrained(self.florence_model_id, trust_remote_code=True).to(self.device).eval()
        
        
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
        raw_frames: (T,H,W,3) uint8 RGB - 원본 이미지
        feature_list: list of (N_objs, h, w) torch tensor (cpu) - 마스크
        """
        color_palette = [
            (255, 0, 0),   # ID 0: Red
            (0, 255, 0),   # ID 1: Green
            (0, 0, 255),   # ID 2: Blue
            (255, 255, 0), # ID 3: Yellow
            (255, 0, 255), # ID 4: Magenta
            (0, 255, 255), # ID 5: Cyan
        ]

        vis = []
        for t in range(len(feature_list)):
            h, w = raw_frames[t].shape[:2]
            
            # 1. 원본 이미지를 배경으로 사용 (0.3을 곱해 어둡게/흐리게 만듦)
            # 이렇게 하면 배경은 은은하게 보이고 마스크 색상이 돋보입니다.
            background = (raw_frames[t].astype(np.float32) * 0.3).astype(np.uint8)
            frame = background.copy()

            masks = feature_list[t] 
            num_objs = masks.shape[0]

            for obj_id in range(num_objs):
                mask = masks[obj_id].numpy()
                mask = (mask > 0.5).astype(np.uint8)

                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                color = color_palette[obj_id % len(color_palette)]
                
                # 2. 마스크가 있는 영역에 색상 입히기 (Alpha Blending)
                # 마스크가 1인 곳에만 색상을 입힙니다.
                for c in range(3): # R, G, B
                    # 배경 이미지 위에 70% 농도의 색상을 덧씌움
                    frame[..., c] = np.where(mask == 1, 
                                            np.clip(background[..., c] + color[c] * 0.7, 0, 255), 
                                            frame[..., c])
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
            success_status = self.process_traj(video_paths, instruction, self.new_path, traj_id=traj_id, data_type=data_type, size=self.size, rgb_skip=self.skip)
            if success_status:
                pass
        except Exception as e:
            print(f"Error processing trajectory {traj_id}: {e}")
            return 0
    
        return 0

    def process_traj(self, video_paths, instruction, save_root,traj_id=0, data_type='val', size=(192,320), rgb_skip=3):
        
        if self.extra_feature == "seg2":
            text_prompt_base = str(instruction).lower().strip()
            if not text_prompt_base or text_prompt_base == "none":
                print(f"Skipping Traj {traj_id}: Empty instruction.")
                return False
            
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
                    num_frames = raw_frames.shape[0]

                    # ---------- 1) dump frames to a temp directory ----------
                    tmp_dir = os.path.join("dataset_example/seg2_tmp", "sam2_frames", f"traj_{traj_id:06d}", f"cam_{video_id}")
                    os.makedirs(tmp_dir, exist_ok=True)

                    for i in range(num_frames):
                        bgr = cv2.cvtColor(raw_frames[i], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(tmp_dir, f"{i}.jpg"), bgr)

                    # ---------- 2) init_state(video_path=...) ----------
                    inference_state = self.video_predictor.init_state(video_path=tmp_dir)

                    # ---------- 3) Grounding DINO on first frame ----------
                    first_frame_pil = Image.fromarray(raw_frames[0]).convert("RGB")
                    W, H = first_frame_pil.size


                    #Dino
                    # text_prompt = str(instruction).lower().strip()
                    # if text_prompt.endswith("."):
                    #     text_prompt = text_prompt[:-1] 

                    # text_prompt += " <and> gripper."
                    # print(f"Final Prompt: {text_prompt}")


                    # transform = GD_T.Compose([
                    #     GD_T.RandomResize([800], max_size=1333),
                    #     GD_T.ToTensor(),
                    #     GD_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    # ])
                    # image_transformed, _ = transform(first_frame_pil, None)

                    # boxes, logits, phrases = predict(
                    #     model=self.grounding_model,
                    #     image=image_transformed.to(self.device),
                    #     caption=text_prompt,
                    #     box_threshold=self.box_threshold,
                    #     text_threshold=self.text_threshold,
                    #     device=self.device
                    # )

                    # W, H = first_frame_pil.size
                    # pixel_boxes = boxes * torch.Tensor([W, H, W, H])
                    # input_boxes = box_cxcywh_to_xyxy(pixel_boxes).detach().cpu().numpy().astype(np.float32)

                    #Florence-2
                    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
                    instruction_raw = str(instruction).lower().strip() 

                    stop_words = r'\b(in|to|on|at|with|near|under|from|and|then)\b'

                    refined_text = re.sub(stop_words, '<and>', instruction_raw)
                    refined_text = re.sub(r'(<and>\s*)+', ' <and> ', refined_text).strip()
                    refined_text = re.sub(r'\b(put|move|pick|place|take|it|left|right|up|down|them|close)\b', '', refined_text).strip()
                    if refined_text.endswith("."): refined_text = refined_text[:-1]

                    
                    if refined_text.endswith("."):
                        refined_text = refined_text[:-1]

                    text_input = f"{refined_text} <and> the gripper."
                    full_prompt = task_prompt + text_input

                    print(full_prompt)

                    # 모델 입력 준비
                    inputs = self.florence_processor(text=full_prompt, images=first_frame_pil, return_tensors="pt").to(self.device)

                    # 추론 실행
                    generated_ids = self.florence_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3
                    )

                    # 결과 해석
                    generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed_answer = self.florence_processor.post_process_generation(
                        generated_text, 
                        task=task_prompt, 
                        image_size=(W, H)
                    )

                    print(parsed_answer)

                    res = parsed_answer.get(task_prompt, {})
                    input_boxes = np.array(res.get('bboxes', [])).astype(np.float32)

                    if len(input_boxes) == 0:
                        print(f"[-] Skipping Traj {traj_id} Cam {video_id}: Florence-2 detected nothing.")
                        self.video_predictor.reset_state(inference_state)
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        return False
                        
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
                            m = (out_mask_logits > 0.0).float()
                            # m: (N,1,H,W) or (N,H,W)
                            if m.ndim == 4:
                                m = m.squeeze(1)
                            masks_per_id = m 
                        else:
                            masks_per_id = torch.zeros((1, H, W), device=self.device)
                            print("detected nothing")

                        if size is not None:
                            masks_per_id = F.interpolate(
                                masks_per_id.unsqueeze(1),  # (N,1,H,W)
                                size=size,
                                mode="nearest"
                            ).squeeze(1)  # (N,H,W)

                        video_segments[out_frame_idx] = masks_per_id.detach().to("cpu")

                    # ---------- 6) ensure all frames exist ----------
                    feature_list = []
                    num_objects = video_segments[0].shape[0] if 0 in video_segments else 1
                    for i in range(num_frames):
                        if i in video_segments:
                            feature_list.append(video_segments[i])
                        else:
                            z = torch.zeros((num_objects, size[0], size[1]))
                            feature_list.append(z)

                    self.video_predictor.reset_state(inference_state)

                    #====Visualization=====
                    save_path = os.path.join(
                        "dataset_example", "outputs", "seg2",
                        data_type, f"{traj_id:06d}", f"{video_id}.mp4")
                    self._save_overlay_video(raw_frames, feature_list, save_path, fps=5, boxes_xyxy=input_boxes)

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
            save_dir = os.path.join(save_root, "libero", self.extra_feature, data_type, str(traj_id))
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{video_id}.pt")
            torch.save(final_features, save_path)
            print(f"[+] Saved: {save_path}")
        return True

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