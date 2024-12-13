from transformers import Pipeline
from PIL import Image
from typing import Union, List, Dict, Any
from .feature_extractor import CLIPFeatureExtractor
from .model import SVIPerceptModel, SVIPerceptConfig

class SVIPerceptPipeline(Pipeline):
    def __init__(self, model=None, **kwargs):
        if model is None:
            # Initialize default model
            config = SVIPerceptConfig()
            model = SVIPerceptModel(config)

        super().__init__(
            model=model,
            feature_extractor=CLIPFeatureExtractor(
                device=kwargs.get('device', None)
            ),
            **kwargs
        )

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> List[Image.Image]:
        """Convert inputs to PIL Images"""
        if isinstance(inputs, str):
            inputs = Image.open(inputs).convert('RGB')
        if isinstance(inputs, Image.Image):
            inputs = [inputs]
        elif isinstance(inputs, list):
            inputs = [
                Image.open(x).convert('RGB') if isinstance(x, str) else x
                for x in inputs
            ]
        return inputs

    def _forward(self, model_inputs: List[Image.Image]) -> Dict[str, Any]:
        """Extract CLIP features and apply custom processing"""
        # Get CLIP features
        features = self.feature_extractor.extract_features(model_inputs)
        # Run KNN model
        return self.model(features["image_features"])

    def postprocess(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format the outputs into a more readable dict"""
        raw = model_outputs['results']
        results = {}
        for cat_i, cat in enumerate(self.model.categories):
            results[cat] = raw[0,cat_i]

        return { 'results': results }
