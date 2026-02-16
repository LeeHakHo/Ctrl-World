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

base_dir = os.path.dirname(os.path.abspath(__file__))
GSAM_ROOT = os.path.join(base_dir, "gsam")
sys.path.append(GSAM_ROOT)

sys.path.append(os.path.join(GSAM_ROOT, "GroundingDINO"))
sys.path.append(os.path.join(GSAM_ROOT, "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# SAM
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

import spacy


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

def load_dino_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    _ = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model.to(device)
    return model

@torch.no_grad()
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    image = image.to(device)
    outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes  = outputs["pred_boxes"].cpu()[0]             # (nq, 4) cxcywh normalized

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt  = boxes[filt_mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

    #=========



class ExtractFeatureDataset(Dataset): 
    def __init__(self, old_path, new_path, device, size=(192, 320), rgb_skip=3, extra_feature=None):
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.device = device
        self.extra_feature=extra_feature
        self.extra_feature = extra_feature

        if self.extra_feature == "seg":

            #LLM for language instruction filtering
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

            self.config_file = os.path.join(GSAM_ROOT, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
            self.grounded_checkpoint = os.path.join(GSAM_ROOT, "groundingdino_swint_ogc.pth")  # 예시
            self.bert_base_uncased_path = "bert-base-uncased" #os.path.join(GSAM_ROOT, "bert-base-uncased")        # 예시(없으면 None 가능)

            self.sam_version = "vit_h"
            self.sam_checkpoint = os.path.join(GSAM_ROOT, "sam_vit_h_4b8939.pth")             # 예시
            self.use_sam_hq = False
            self.sam_hq_checkpoint = None

            self.box_threshold = 0.3
            self.text_threshold = 0.25

            # 1) DINO
            self.dino_model = load_dino_model(
                self.config_file,
                self.grounded_checkpoint,
                self.bert_base_uncased_path,
                device=device,
            )

            # 2) SAM predictor
            if self.use_sam_hq:
                sam = sam_hq_model_registry[self.sam_version](checkpoint=self.sam_hq_checkpoint)
            else:
                sam = sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint)
            sam.to(device).eval()
            self.predictor = SamPredictor(sam)
            

        def load_json_file(file_path):
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))  # 使用 json.loads() 解析单行
            return data

        self.data = load_json_file(f'{old_path}/meta/episodes.jsonl')

    #======segmenation model =======

    @torch.no_grad()
    def frame_to_mask(self, frame_chw_float_m11: torch.Tensor, text_prompt: str, device):
        #text_prompt = text_prompt.rstrip(".") + ". gripper." #if you want segment gripper
        
        # 1) frame -> RGB uint8
        frame = (frame_chw_float_m11 * 0.5 + 0.5).clamp(0, 1)
        frame_rgb = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # H,W,3 RGB

        # 2) DINO
        image_pil, image_tensor = load_image_from_rgb(frame_rgb)

        # 3) DINO
        boxes_filt, pred_phrases = get_grounding_output(
            self.dino_model,
            image_tensor,
            text_prompt,
            self.box_threshold,
            self.text_threshold,
            device=device,
        )

        # 4) cxcywh normalized -> xyxy pixel
        H, W = frame_rgb.shape[:2]
        if boxes_filt.numel() > 0:
            boxes = boxes_filt.clone()
            boxes = boxes * torch.tensor([W, H, W, H])
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]
            boxes_xyxy = boxes
        else:
            boxes_xyxy = boxes_filt

        # 5) SAM mask
        self.predictor.set_image(frame_rgb)
        if boxes_xyxy.numel() == 0:
            # 0 mask if you can't find anything
            mask = torch.zeros((1, H, W), dtype=torch.float32, device=device)
        else:
            transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_xyxy, (H, W)).to(device)
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=True,
            )
            # masks: (N,1,H,W) -> (1,H,W)
            mask = (masks[:, 0].float().sum(dim=0, keepdim=True) > 0).float()  # (1,H,W)


        #=====Visualization for checking====
        import os

        # 디버그용: 프레임을 계속 모아서 한 번에 비디오로 저장
        if (not hasattr(self, "_debug_frames")):
            self._debug_frames = []
            self._debug_video_saved = False

        # 저장할 최대 프레임 수 (너무 커지면 메모리 터짐)
        MAX_DEBUG_FRAMES = 240   # 5fps면 약 24초
        DEBUG_FPS = 5            # 너 rgb_skip=3이면 대략 5Hz라서 5 권장

        # text_prompt가 있고, 아직 프레임을 더 모을 수 있을 때만 append
        if bool(text_prompt.strip()) and (len(self._debug_frames) < MAX_DEBUG_FRAMES):
            # mask overlay 만들기 (빨간 채널)
            mask_np = (mask[0].detach().cpu().numpy() > 0.5).astype(np.uint8)  # (H,W) 0/1
            overlay = frame_rgb.copy()
            overlay[..., 0] = np.maximum(overlay[..., 0], mask_np * 255)

            # bbox도 영상에 그려 넣고 싶으면, cv2로 직접 그리는 게 제일 빠름
            # (matplotlib로 프레임마다 그리면 너무 느림)
            try:
                import cv2
                if boxes_xyxy.numel() > 0:
                    for box in boxes_xyxy:
                        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간 박스 (BGR)
            except ImportError:
                pass

            self._debug_frames.append(overlay)

        # 프레임을 다 모았으면 한 번만 저장
        if (len(self._debug_frames) >= MAX_DEBUG_FRAMES) and (not self._debug_video_saved):
            self._debug_video_saved = True

            save_dir = os.path.join("dataset_example/outputs/", "debug_vis")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, text_prompt + ".mp4")

            import mediapy as media
            media.write_video(save_path, self._debug_frames, fps=DEBUG_FPS)
            print(f"[DEBUG] Saved video to {save_path} (frames={len(self._debug_frames)}, fps={DEBUG_FPS})", flush=True)

        # ================================================




        return mask

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
            frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0*2-1
            frames = frames[::rgb_skip]  # Skip frames to save memory here!!!
            with torch.no_grad():
                batch_size = 16
                feature_list = []
                for i in range(0, len(frames), batch_size):
                    batch = frames[i:i+batch_size].to(self.device)


                    for b in range(batch.shape[0]):
                        if self.extra_feature == "seg":
                            #print("use feature segmentation")
                            mask = self.frame_to_mask(batch[b], text_prompt=instruction, device=self.device)  # (4,H_lat,W_lat)
                            
                            if size is not None:
                                mask = torch.nn.functional.interpolate(
                                    mask.unsqueeze(0), size=size, mode='nearest'
                                ).squeeze(0)
                            
                            feature_list.append(mask.cpu())
                            
            final_features = torch.stack(feature_list, dim=0)
            os.makedirs(f"{save_root}/latent_videos/{self.extra_feature}/{data_type}/{traj_id}", exist_ok=True)
            torch.save(final_features, f"{save_root}/latent_videos/{self.extra_feature}/{data_type}/{traj_id}/{video_id}.pt")

if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--droid_hf_path', type=str, default='/scr/shared/world_model/DROID-1.0.1')
    parser.add_argument('--droid_output_path', type=str, default='/scr/hyeonhoo/outputs/extract_latent/')
    parser.add_argument('--extra_feature', type=str, default=None)
    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    accelerator = Accelerator()
    dataset = ExtractFeatureDataset(
        old_path=args.droid_hf_path,
        new_path= args.droid_output_path,
        device=accelerator.device,
        rgb_skip=3, #  to downsample 15hz video to 5hz video
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