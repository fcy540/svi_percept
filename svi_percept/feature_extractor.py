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
        images: Union[Image.Image, List[Image.Image]]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract CLIP image features from one or more images.

        Args:
            images: Single PIL Image or list of PIL Images

        Returns:
            dict: Contains 'image_features' tensor of shape (N, feature_dim)
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Preprocess images
        processed_images = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)

        # Extract features
        image_features = self.model.encode_image(processed_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return {"image_features": image_features}

