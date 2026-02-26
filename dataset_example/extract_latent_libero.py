"""
Process lerobot/libero_10_image dataset into Ctrl-World required format

LIBERO dataset structure:
- observation.images.image: Main view image (dict with 'bytes' and 'path')
- observation.images.wrist_image: Wrist view image
- observation.state: 8-dim state (7 joint angles + 1 gripper)
- action: 7-dim action (6-DOF end-effector delta + gripper)
- meta/tasks.parquet: Task descriptions (index is the task description text)
- meta/episodes/: Metadata for each episode (contains task descriptions)

Important Note:
===============
This script directly uses LIBERO's native joint angle representation without Cartesian coordinate conversion.

Reason:
1. When training from scratch, the model can learn any state representation (joint angles, Cartesian coordinates, etc.)
2. Field names in training code (like 'observation.state.cartesian_position') are just labels
   The actual data format is defined by us
3. LIBERO's 8-dim joint angles vs DROID's 7-dim Cartesian coordinates:
   - Similar dimensions (8 vs 7)
   - Both can fully represent robot state
   - Model will adapt when training from scratch

If you need to use pretrained DROID model:
- Need to implement Forward Kinematics (FK) to convert joint angles to Cartesian coordinates
- Refer to get_fk_solution function in models/utils.py
"""

import mediapy
import os
from diffusers.models import AutoencoderKLTemporalDecoder
import torch
import numpy as np
import json
from torch.utils.data import Dataset
import pandas as pd
from accelerate import Accelerator
from PIL import Image
import io
from tqdm import tqdm


class EncodeLatentDatasetLibero(Dataset): 
    def __init__(self, libero_hf_path, new_path, svd_path, device, size=(192, 320), fps=5, frame_skip=1, flip_horizontal=False, use_cached=False):
        """
        Args:
            libero_hf_path: HuggingFace cache path for LIBERO dataset
            new_path: Output path
            svd_path: SVD model path
            device: Device
            size: Video resolution (height, width), None means no resize
            fps: Output video frame rate
            frame_skip: Frame skip interval (e.g., frame_skip=2 means take 1 frame every 2 frames)
            flip_horizontal: Whether to flip images horizontally (correct left-right direction)
            use_cached: Whether to use cached data
        """
        self.libero_hf_path = libero_hf_path
        self.new_path = new_path
        self.size = size
        self.fps = fps
        self.frame_skip = frame_skip
        self.flip_horizontal = flip_horizontal
        self.use_cached = use_cached
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)

        # Find dataset snapshot path
        # snapshot_dir = os.path.join(libero_hf_path, 'snapshots')
        # snapshot_id = os.listdir(snapshot_dir)[0]
        # self.data_root = os.path.join(snapshot_dir, snapshot_id)
        self.data_root = libero_hf_path
        print(f"Data root directory: {self.data_root}")
        
        # Load all episode data files
        data_dir = os.path.join(self.data_root, 'data/chunk-000')
        self.data_files = sorted([
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.parquet')
        ])
        
        # ----------------------------
        # Load task information
        # ----------------------------
        self.task_descriptions = {}
        self.num_tasks = 0

        tasks_jsonl = os.path.join(self.data_root, "meta", "tasks.jsonl")
        if os.path.exists(tasks_jsonl):
            # tasks.jsonl: each line is a JSON object.
            # We'll build task_index -> task description mapping.
            with open(tasks_jsonl, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)

                    # Try common fields
                    # Some datasets store: {"task_index": 0, "task": "..."} or {"id":0,"text":"..."}
                    task_idx = obj.get("task_index", obj.get("id", obj.get("index", None)))
                    task_text = obj.get("task", obj.get("text", obj.get("description", None)))

                    if task_idx is not None and task_text is not None:
                        self.task_descriptions[int(task_idx)] = str(task_text)

            self.num_tasks = len(self.task_descriptions)
            print(f"Loaded {self.num_tasks} tasks from {tasks_jsonl}")
        else:
            print(f"⚠️ tasks.jsonl not found: {tasks_jsonl}")

        # ----------------------------
        # Load episode -> task description mapping
        # ----------------------------
        self.episode_tasks = {}

        episodes_jsonl = os.path.join(self.data_root, "meta", "episodes.jsonl")
        if os.path.exists(episodes_jsonl):
            with open(episodes_jsonl, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)

                    # Try common fields
                    ep_idx = obj.get("episode_index", obj.get("episode_id", obj.get("id", None)))

                    # tasks could be list of strings or list of dicts
                    tasks = obj.get("tasks", None)
                    task_text = None
                    if isinstance(tasks, list) and len(tasks) > 0:
                        if isinstance(tasks[0], str):
                            task_text = tasks[0]
                        elif isinstance(tasks[0], dict):
                            task_text = tasks[0].get("task", tasks[0].get("text", tasks[0].get("description", None)))

                    # sometimes it's stored directly
                    if task_text is None:
                        task_text = obj.get("task", obj.get("text", obj.get("description", None)))

                    if ep_idx is not None and task_text is not None:
                        self.episode_tasks[int(ep_idx)] = str(task_text)

            print(f"Loaded task descriptions for {len(self.episode_tasks)} episodes from {episodes_jsonl}")
        else:
            print(f"⚠️ episodes.jsonl not found: {episodes_jsonl}")
        
        # Collect all episodes
        self.episodes = []
        for data_file in self.data_files:
            df = pd.read_parquet(data_file)
            unique_episodes = df['episode_index'].unique()
            for ep_idx in unique_episodes:
                ep_data = df[df['episode_index'] == ep_idx]
                task_idx = ep_data['task_index'].iloc[0]
                self.episodes.append({
                    'file': data_file,
                    'episode_index': int(ep_idx),
                    'task_index': int(task_idx),
                    'length': len(ep_data)
                })
        
        print(f"Total {len(self.episodes)} episodes")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode_info = self.episodes[idx]
        episode_idx = episode_info['episode_index']
        
        # Check if already processed
        data_type = 'val' if episode_idx % 100 == 99 else 'train'
        if self.use_cached and os.path.exists(f"{self.new_path}/annotation/{data_type}/{episode_idx}.json"):
            return 0
        
        try:
            # Load all data for this episode
            df = pd.read_parquet(episode_info['file'])
            ep_data = df[df['episode_index'] == episode_idx].reset_index(drop=True)
            
            # Extract task description (read from metadata)
            instruction = self._get_task_instruction(episode_idx, episode_info['task_index'])
            
            # Process trajectory
            self.process_traj(ep_data, instruction, self.new_path, 
                            episode_idx=episode_idx, data_type=data_type, 
                            size=self.size, device=self.vae.device)
        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
        return 0

    def _get_task_instruction(self, episode_idx, task_idx):
        """
        Get task description
        Prioritize reading from episodes metadata (more accurate), otherwise read from tasks.parquet
        
        Args:
            episode_idx: episode index
            task_idx: task index
        """
        # Prioritize using task description read from episodes metadata
        if episode_idx in self.episode_tasks:
            return self.episode_tasks[episode_idx]
        
        # Fallback: use task description from tasks.parquet
        if task_idx in self.task_descriptions:
            return self.task_descriptions[task_idx]
        
        # Final fallback
        return f"Task {task_idx}"

    def _decode_image(self, img_dict):
        """Decode image from dictionary"""
        if 'bytes' in img_dict and img_dict['bytes']:
            img = Image.open(io.BytesIO(img_dict['bytes']))
            # Flip horizontally if needed (correct left-right direction)
            if self.flip_horizontal:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return np.array(img)
        elif 'path' in img_dict and img_dict['path']:
            # Load from path (if needed)
            img_path = os.path.join(self.libero_hf_path, img_dict['path'])
            if os.path.exists(img_path):
                img = Image.open(img_path)
                # Flip horizontally if needed (correct left-right direction)
                if self.flip_horizontal:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                return np.array(img)
        raise ValueError("Unable to decode image")

    def process_traj(self, ep_data, instruction, save_root, episode_idx=0, 
                    data_type='val', size=(192, 320), device='cuda'):
        """
        Process single trajectory and save
        
        Args:
            ep_data: DataFrame containing all frames for this episode
            instruction: Task description
            save_root: Save root directory
            episode_idx: episode index
            data_type: 'train' or 'val'
            size: Video resolution
            device: Device
        """
        length = len(ep_data)
        
        # Collect images and data (Note: collect all frames here, downsample later)
        images_main = []
        images_wrist = []
        states = []
        actions = []
        
        for i in range(length):
            row = ep_data.iloc[i]
            
            # Decode images
            img_main = self._decode_image(row['observation.images.image'])
            img_wrist = self._decode_image(row['observation.images.wrist_image'])
            
            images_main.append(img_main)
            images_wrist.append(img_wrist)
            
            # States and actions
            states.append(row['observation.state'].tolist())
            actions.append(row['action'].tolist())
        
        # ⚠️ Critical: frame downsampling (consistent with DROID)
        # Example: frame_skip=2 means take 1 frame every 2 frames, thus reducing frame rate
        if self.frame_skip > 1:
            images_main = images_main[::self.frame_skip]
            images_wrist = images_wrist[::self.frame_skip]
            states = states[::self.frame_skip]
            actions = actions[::self.frame_skip]
        
        # LIBERO has two views: main view + wrist view
        # To be compatible with Ctrl-World (requires 3 views), we duplicate main view as the third view
        video_sequences = [
            images_main,   # View 0: Main view
            images_wrist,  # View 1: Wrist view  
            images_main,   # View 2: Main view copy (for compatibility)
        ]
        
        # Process video for each view
        for video_id, video_frames in enumerate(video_sequences):
            # Convert to tensor
            frames_array = np.stack(video_frames)  # (T, H, W, C)
            frames = torch.tensor(frames_array).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1  # (T, C, H, W)
            
            # Resize (optional)
            if size is not None:
                x = torch.nn.functional.interpolate(frames, size=size, mode='bilinear', align_corners=False)
            else:
                x = frames  # No resize, keep original size
            
            # Save resized video
            resize_video = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
            resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            
            os.makedirs(f"{save_root}/videos/{data_type}/{episode_idx}", exist_ok=True)
            mediapy.write_video(
                f"{save_root}/videos/{data_type}/{episode_idx}/{video_id}.mp4", 
                resize_video, 
                fps=self.fps
            )

            # Encode and save latent representation
            x = x.to(device)
            with torch.no_grad():
                # Adjust batch_size based on image size
                # Larger original size → smaller batch_size (avoid GPU memory explosion)
                if self.size is None or (self.size[0] * self.size[1] > 192 * 320):
                    batch_size = 16  # Small batch for large size
                else:
                    batch_size = 64  # Standard size
                
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = self.vae.encode(batch).latent_dist.sample().mul_(self.vae.config.scaling_factor).cpu()
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            
            os.makedirs(f"{save_root}/latent_videos/{data_type}/{episode_idx}", exist_ok=True)
            torch.save(x, f"{save_root}/latent_videos/{data_type}/{episode_idx}/{video_id}.pt")
        
        # LIBERO native format:
        # - state: 8-dim (7 joint angles + 1 gripper)
        # - action: 7-dim (6-DOF end-effector delta + gripper)
        
        # When training from scratch, directly use LIBERO's joint angle representation!
        # 'observation.state.cartesian_position' in training code is just a field name
        # It can actually store any state representation (joint angles, Cartesian coordinates, etc.)
        
        # Construct state: directly use LIBERO's joint angles
        # states is already 8-dim: [joint1, joint2, ..., joint7, gripper]
        robot_states = []
        for state in states:
            # Directly use first 7 joint angles + gripper
            robot_states.append(state.tolist() if hasattr(state, 'tolist') else state)
        
        # Save annotation file
        info = {
            "texts": [instruction],
            "episode_id": episode_idx,  # Note: use episode_id instead of episode_index (consistent with DROID)
            "success": 1,  # LIBERO dataset assumes all are successful
            "video_length": len(robot_states),  # Length after downsampling
            "state_length": len(robot_states),  # Length after downsampling
            "raw_length": length,  # Original length (before downsampling)
            "videos": [
                {"video_path": f"videos/{data_type}/{episode_idx}/0.mp4"},
                {"video_path": f"videos/{data_type}/{episode_idx}/1.mp4"},
                {"video_path": f"videos/{data_type}/{episode_idx}/2.mp4"}
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{episode_idx}/0.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{episode_idx}/1.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{episode_idx}/2.pt"}
            ],
            # Core: directly use LIBERO's joint angles (8-dim) as state
            # Field name is 'cartesian_position' but actually stores joint angles
            'states': robot_states,  # Complete 8-dim state
            'observation.state.cartesian_position': [[s[0], s[1], s[2], s[3], s[4], s[5], s[6]] for s in robot_states],  # First 7 dims joints
            'observation.state.joint_position': robot_states,  # Complete state
            'observation.state.gripper_position': [s[-1] for s in robot_states],  # 8th dim gripper
            
            # Action: LIBERO action is 7-dim (6-DOF delta + gripper)
            'action.cartesian_position': [[a[0], a[1], a[2], a[3], a[4], a[5]] for a in actions],  # First 6 dims
            'action.joint_position': actions,  # Complete action
            'action.gripper_position': [a[-1] for a in actions],  # 7th dim gripper
            'action.joint_velocity': [[0.0] * 7 for _ in actions],  # Placeholder (not used in training)
        }
        
        os.makedirs(f"{save_root}/annotation/{data_type}", exist_ok=True)
        with open(f"{save_root}/annotation/{data_type}/{episode_idx}.json", "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--libero_hf_path', type=str, 
                       default='/scr2/shared/world_model/libero/',
                       help='HuggingFace cache path for LIBERO dataset')
    parser.add_argument('--output_path', type=str, default='/scr/hyeonhoo/outputs/libero',
                       help='Save path for processed data')
    parser.add_argument('--svd_path', type=str, default='/scr/hyeonhoo/checkpoints/stable-video-diffusion-img2vid',
                       help='SVD model path')
    parser.add_argument('--video_size', type=str, default='192x320',
                       help='Video resolution (format: HxW, e.g., 192x320 or 256x448, set to "none" for no resize)')
    parser.add_argument('--frame_skip', type=int, default=4,
                       help='Frame downsampling interval (1=no downsampling, 2=take 1 frame every 2 frames, 3=take 1 frame every 3 frames)')
    parser.add_argument('--flip_horizontal', action='store_true',
                       help='Flip images horizontally (correct left-right direction, enable if task description left-right is opposite to video)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (process only small amount of data)')
    parser.add_argument('--use_cached', action='store_true', help='Skip already processed episodes')
    args = parser.parse_args()
    
    # Parse video resolution
    if args.video_size.lower() == 'none':
        video_size = None
        print("⚠️  No resize, using original image size (GPU memory usage will increase)")
    else:
        h, w = map(int, args.video_size.split('x'))
        video_size = (h, w)
        print(f"✅ Video resolution: {h}×{w}")

    accelerator = Accelerator()
    
    if args.flip_horizontal:
        print("⚠️  Horizontal flip enabled (correcting left-right direction)")
    
    dataset = EncodeLatentDatasetLibero(
        libero_hf_path=args.libero_hf_path,
        new_path=args.output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        size=video_size,
        fps=5,
        frame_skip=args.frame_skip,
        flip_horizontal=args.flip_horizontal,
        use_cached=args.use_cached
    )
    
    tmp_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    
    for idx, _ in enumerate(tqdm(tmp_data_loader, disable=not accelerator.is_main_process)):
        if idx == 5 and args.debug:
            print("Debug mode: Processed 5 samples, stopping")
            break
        if idx % 10 == 0 and accelerator.is_main_process:
            print(f"Processed {idx}/{len(dataset)} episodes")

    if accelerator.is_main_process:
        print(f"\n✅ Processing complete! Data saved to: {args.output_path}")
        print(f"   - Videos: {args.output_path}/videos/")
        print(f"   - Latents: {args.output_path}/latent_videos/")
        print(f"   - Annotations: {args.output_path}/annotation/")

# Usage examples:
# 1. Standard configuration (consistent with DROID)
# accelerate launch dataset_example/extract_latent_libero.py --frame_skip 4 --debug
#
# 2. Correct left-right direction (if video left-right is opposite to task description)
# accelerate launch dataset_example/extract_latent_libero.py --frame_skip 4 --flip_horizontal
#
# 3. Higher resolution (preserve more details)
# accelerate launch dataset_example/extract_latent_libero.py --frame_skip 4 --video_size 256x448
#
# 4. No resize (use original size, ⚠️ high GPU memory usage)
# accelerate launch dataset_example/extract_latent_libero.py --frame_skip 4 --video_size none
#
# Parameter explanation:
# --frame_skip: Frame downsampling interval
#   - 1: No downsampling (~20Hz)
#   - 2: Downsample to ~10Hz
#   - 3: Downsample to ~6-7Hz
#   - 4: Downsample to ~5Hz (recommended, consistent with DROID)
#
# --video_size: Video resolution
#   - 192x320: Standard configuration (consistent with DROID)
#   - 256x448: Higher resolution
#   - none: No resize, keep original size (⚠️ GPU memory usage increases by 6x)
#
# --flip_horizontal: Flip images horizontally
#   - Enable this option if task says "put on the left" but object is on the right in video
#   - Common in datasets where robot perspective vs camera perspective is inconsistent
