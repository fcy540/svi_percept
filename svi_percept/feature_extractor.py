#  svi_percept - CLIP-based feature extractor
#  Copyright (C) 2024 Matthew Danish
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import open_clip
from PIL import Image
from typing import Union, List, Dict, Any

class CLIPFeatureExtractor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", model_name: str = None, pretrained: str = None):
        """
        Initialize the CLIP feature extractor using ViT-H-14-378-quickgelu.

        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = "ViT-H-14-378-quickgelu" if model_name is None else model_name
        self.pretrained = "dfn5b" if pretrained is None else pretrained

        # Load model and preprocessing
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device
        )
        self.model.eval()

    @torch.no_grad()
    def extract_features(
        self, 
        images: Union[Image.Image, List[Image.Image], dict, List[dict]]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract CLIP image features from one or more images.

        Args:
            images: Single PIL Image or list of PIL Images

        Returns:
            dict: Contains 'image_features' tensor of shape (N, feature_dim)
        """
        if isinstance(images, dict):
            images = images['image']
        if not isinstance(images, list):
            images = [images]
        else:
            images = [x['image'] if isinstance(x, dict) else x for x in images]

        # Preprocess images
        processed_images = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)

        # Extract features
        image_features = self.model.encode_image(processed_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return {"image_features": image_features}

