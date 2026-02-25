import copy

import timm
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# import torchvision
import logging

from .module_attr_mixin import ModuleAttrMixin

from .pytorch_util import replace_submodules

logger = logging.getLogger(__name__)

def generate_2d_positional_embedding(height, width, d_model):
    assert d_model % 2 == 0, "d_model should be even"

    pe = torch.zeros(height, width, d_model)
    d_model_half = d_model // 2
    div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * -(math.log(10000.0) / d_model_half))
    pos_w = torch.arange(0, width).unsqueeze(1)
    pos_h = torch.arange(0, height).unsqueeze(1)
    pe[:, :, 0:d_model_half:2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
    pe[:, :, 1:d_model_half:2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
    pe[:, :, d_model_half::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)
    pe[:, :, d_model_half+1::2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)
    return pe

def generate_2d_positional_embedding_polarv1(height, width, d_model):
    assert d_model % 4 == 0, "d_model should be divisible by 4"

    pe = torch.zeros(height, width, d_model)
    d_model_half = d_model // 2
    div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * -(math.log(10000.0) / d_model_half))
    div_term_theta = (torch.arange(2, d_model_half+2, 2).float()) * np.pi
    pos_w = torch.arange(0, width).unsqueeze(1).repeat(1, height)
    pos_h = torch.arange(0, height).unsqueeze(0).repeat(width, 1)
    pos_w_centered = (pos_w - width // 2)
    pos_h_centered = (pos_h - height // 2)
    pos_r = torch.sqrt(pos_w_centered**2 + pos_h_centered**2).unsqueeze(2).repeat(1, 1, d_model_half // 2)
    pos_theta = torch.atan2(pos_w_centered, pos_h_centered).unsqueeze(2).repeat(1, 1, d_model_half // 2)
    pos_theta = (pos_theta + np.pi) / (2 * np.pi)
    div_term = div_term.unsqueeze(0).unsqueeze(0).repeat(height, width, 1)
    div_term_theta = div_term_theta.unsqueeze(0).unsqueeze(0).repeat(height, width, 1)
    pe[:, :, 0:d_model_half:2] = torch.sin(pos_r * div_term)
    pe[:, :, 1:d_model_half:2] = torch.cos(pos_r * div_term)
    pe[:, :, d_model_half::2] = torch.sin(pos_theta * div_term_theta)
    pe[:, :, d_model_half+1::2] = torch.cos(pos_theta * div_term_theta)
    return pe

def generate_2d_positional_embedding_polarv2(height, width, d_model):
    assert d_model == 768, "d_model should be 768"

    pe = torch.zeros(height, width, d_model)
    d_model_half = d_model // 2
    div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * -(math.log(10000.0) / d_model_half))
    div_term_theta = torch.zeros(d_model_half // 2)
    for i in range(8):
        div_term_theta[i::8] = (torch.arange(0, d_model_half // 8, 2).float()) * np.pi
    pos_w = torch.arange(0, width).unsqueeze(1).repeat(1, height)
    pos_h = torch.arange(0, height).unsqueeze(0).repeat(width, 1)
    pos_w_centered = (pos_w - width // 2)
    pos_h_centered = (pos_h - height // 2)
    pos_r = torch.sqrt(pos_w_centered**2 + pos_h_centered**2).unsqueeze(2).repeat(1, 1, d_model_half // 2)
    pos_theta = torch.atan2(pos_w_centered, pos_h_centered).unsqueeze(2).repeat(1, 1, d_model_half // 2)
    pos_theta = (pos_theta + np.pi) / (2 * np.pi)
    div_term = div_term.unsqueeze(0).unsqueeze(0).repeat(height, width, 1)
    div_term_theta = div_term_theta.unsqueeze(0).unsqueeze(0).repeat(height, width, 1)
    pe[:, :, 0:d_model_half:2] = torch.sin(pos_r * div_term)
    pe[:, :, 1:d_model_half:2] = torch.cos(pos_r * div_term)
    pe[:, :, d_model_half::2] = torch.sin(pos_theta * div_term_theta)
    pe[:, :, d_model_half+1::2] = torch.cos(pos_theta * div_term_theta)
    return pe

class ObjectCentricPool2d(nn.Module):
    def __init__(self, n_emb: int, height: int, width: int, positional_embedding: str):
        super(ObjectCentricPool2d, self).__init__()
        self.height = height
        self.width = width
        if positional_embedding == 'sine':
            self.positional_embedding = generate_2d_positional_embedding(height, width, n_emb)
        elif positional_embedding == 'polarv1':
            self.positional_embedding = generate_2d_positional_embedding_polarv1(height, width, n_emb)
        elif positional_embedding == 'polarv2':
            self.positional_embedding = generate_2d_positional_embedding_polarv2(height, width, n_emb)
        else:
            raise RuntimeError(f"Unsupported positional embedding: {positional_embedding}")
        logger.info(f"Using `{positional_embedding}` positional embedding in ObjectCentricPool2d")
        self.global_object_embedding = nn.Parameter(torch.randn(n_emb))
        self.empty_object_embedding = nn.Parameter(torch.randn(n_emb))
        self.n_emb = n_emb
    
    def forward(self, x):
        """
            x: B, num_objs, H, W (dtype=torch.bool)
            return: B, n_emb
        """
        bs = x.shape[0]
        y_coords = torch.arange(self.height, device=x.device).view(1, -1, 1).expand(bs, -1, self.width)
        x_coords = torch.arange(self.width, device=x.device).view(1, 1, -1).expand(bs, self.height, -1)
        true_y_sum = (y_coords * x).sum(dim=(1, 2)).float()
        true_x_sum = (x_coords * x).sum(dim=(1, 2)).float()
        true_count = x.sum(dim=(1, 2))
        # consider that the true count would be 0
        true_y = torch.where(true_count > 0, true_y_sum / true_count, 0).int()
        true_x = torch.where(true_count > 0, true_x_sum / true_count, 0).int()
        pos_emb = self.positional_embedding.to(x.device)[true_y, true_x]
        global_emb = self.global_object_embedding.expand(bs, -1)
        object_emb = torch.where((true_count > 0).unsqueeze(1).repeat(1, self.n_emb), global_emb + pos_emb, self.empty_object_embedding)
        return object_emb

class GrayScaleFeatureExtractor(nn.Module):
    def __init__(self, n_dim: int, height: int, width: int, target_size=(32, 32), threshold=0.5):
        super(GrayScaleFeatureExtractor, self).__init__()
        self.target_size = target_size
        self.threshold = threshold
        self.positional_embedding = generate_2d_positional_embedding(height, width, n_dim)

        # Define the CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Global max pooling to reduce to a fixed size
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(64, n_dim)

    def get_batch_bounding_boxes(self, batch_images):
        """
        Batch version to get the bounding boxes for each image in the batch.
        Args:
            batch_images (torch.Tensor): Input tensor of shape (batch_size, 1, H, W)
        Returns:cam
            bboxes (list of tuples): A list of bounding boxes (xmin, ymin, xmax, ymax) for each image.
        """
        batch_size = batch_images.size(0)
        bboxes = []

        for i in range(batch_size):
            image = batch_images[i, 0]  # Shape (H, W) for each image in the batch
            binary_mask = image > self.threshold
            nonzero_indices = torch.nonzero(binary_mask)

            if nonzero_indices.numel() == 0:
                # If no pixels are above the threshold, use the full image
                bbox = (0, 0, 1, 1)
            else:
                ymin, xmin = torch.min(nonzero_indices, dim=0)[0]
                ymax, xmax = torch.max(nonzero_indices, dim=0)[0]
                ymin = ymin.item()
                xmin = xmin.item()
                ymax = ymax.item()
                xmax = xmax.item()
                # consider the case where the bounding box is a line or a point, expand it
                if ymin == ymax:
                    ymin = max(0, ymin - 1)
                    ymax = min(image.size(0), ymax + 1)
                if xmin == xmax:
                    xmin = max(0, xmin - 1)
                    xmax = min(image.size(1), xmax + 1)
                bbox = (xmin, ymin, xmax, ymax)

            bboxes.append(bbox)

        return bboxes

    def crop_and_resize_batch(self, batch_images, bboxes):
        """
        Batch version to crop and resize each image to its corresponding bounding box.
        Args:
            batch_images (torch.Tensor): Input tensor of shape (batch_size, 1, H, W)
            bboxes (list of tuples): List of bounding boxes (xmin, ymin, xmax, ymax) for each image.
        Returns:
            resized_batch (torch.Tensor): Batch of resized images of shape (batch_size, 1, target_size[0], target_size[1])
        """
        resized_images = []
        resize_transform = transforms.Resize(self.target_size)

        for i, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = bbox
            cropped_image = batch_images[i, :, ymin:ymax, xmin:xmax]  # Crop to bounding box
            resized_image = resize_transform(cropped_image)  # Resize to target size
            resized_images.append(resized_image)

        return torch.stack(resized_images)

    def forward(self, x):
        # process x if not batch
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # Step 1: Get bounding boxes for the batch
        bboxes = self.get_batch_bounding_boxes(x)
        # Step 2: Crop and resize the images to the target size (e.g., 64x64)
        x = self.crop_and_resize_batch(x, bboxes)
        # Step 3: Apply the CNN layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Step 4: Global max pooling and fully connected layer
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)
        x = self.fc(x)
        bboxes_shape_y = [b[3] - b[1] for b in bboxes]
        bboxes_shape_x = [b[2] - b[0] for b in bboxes]
        pos_emb = self.positional_embedding.to(x.device)[bboxes_shape_y, bboxes_shape_x]
        x = x + pos_emb
        return x

class OCObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            global_pool: str='',
            n_emb: int=768,
            feature_aggregation: str=None,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_model_map_local = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()
        
        obs_shape_meta = shape_meta['control_obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            num_objs = attr['num_objs']
            type = attr.get('type', 'low_dim')
            positional_embedding = attr.get('positional_embedding', 'sine')
            local_feature = attr.get('local_feature', None)
            logger.warning(f"Using `{positional_embedding}` positional embedding in OCObsEncoder")
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                key_model_map[key] = nn.ModuleDict()
                key_model_map_local[key] = nn.ModuleDict()
                for k in range(num_objs):
                    key_model_map[key][str(k)] = ObjectCentricPool2d(n_emb=n_emb, height=shape[0], width=shape[1], positional_embedding=positional_embedding).to(self.device)
                    if local_feature is not None:
                        if local_feature == 'graycnn':
                            key_model_map_local[key][str(k)] = GrayScaleFeatureExtractor(n_dim=n_emb, height=shape[0], width=shape[1]).to(self.device)
                        else:
                            raise RuntimeError(f"Unsupported local feature: {local_feature}")
                proj = nn.Identity()
                # if feature_size != n_emb:
                #     proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj
            elif type == 'low_dim':
                raise NotImplementedError()
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_model_map_local = key_model_map_local
        self.key_projection_map = key_projection_map
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        raise NotImplementedError()
        
    def forward(self, obs_dict):
        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            assert key == 'camera0_rgb_narrow_objs'
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            embs = []
            for k in range(T):
                raw_feature = self.key_model_map[key][str(k)](img[:, k])
                emb = self.key_projection_map[key](raw_feature)
                embs.append(emb)
                if key in self.key_model_map_local and str(k) in self.key_model_map_local[key]:
                    local_feature = self.key_model_map_local[key][str(k)](img[:, k])
                    embs.append(local_feature)
            embs = torch.stack(embs, dim=1)
            embeddings.append(embs)

        # process lowdim input
        for key in self.low_dim_keys:
            raise NotImplementedError()
        
        # concatenate all features along t
        result = torch.cat(embeddings, dim=1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['control_obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            # dtype = eval(attr['dtype']) if 'dtype' in attr else self.dtype
            dtype = self.dtype
            assert attr['horizon'] == 1
            this_obs = torch.zeros(
                (1, attr['num_objs']) + shape, 
                dtype=dtype,
                device=self.device)
            this_obs[0, 0, :100, :50] = 1.
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 3
        assert example_output.shape[0] == 1

        return example_output.shape


def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize('../diffusion_policy/config'):
        cfg = hydra.compose('train_diffusion_transformer_umi_workspace')
        OmegaConf.resolve(cfg)

    shape_meta = cfg.task.shape_meta
    encoder = TransformerObsEncoder(
        shape_meta=shape_meta
    )