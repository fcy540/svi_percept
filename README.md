# SVI Percept

A Python package for image perception using CLIP features and custom scoring matrices.

## Installation

```bash
pip install .
```

## Quick Start

```python
from svi_percept import SVIPerceptPipeline
from PIL import Image

# Initialize pipeline
pipeline = SVIPerceptPipeline()

# Process single image
image = Image.open("example.jpg").convert('RGB')
results = pipeline(image)
```

or
```python
results = pipeline("example.jpg")
```

The [default model](https://huggingface.co/Spatial-Data-Science-and-GEO-AI-Lab/svi_percept) is based on an Amsterdam case study.

## Features

- K-nearest-neighbour model of human perception
- Batch processing support
- GPU acceleration

## Detailed Usage

### Single Image Processing
```python
from svi_percept import SVIPerceptPipeline

pipeline = SVIPerceptPipeline()
output = pipeline("path/to/image.jpg")

for cat in ['walkability', 'bikeability', 'pleasantness', 'greenness', 'safety']:
    print(f'{cat} score = {output['results'][cat]}')

```

### Batch Processing
```python
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        return {"image": self.image_paths[idx]}

# Process multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
dataset = ImageDataset(image_paths)
pipeline = SVIPerceptPipeline(batch_size=32)
outputs = pipeline(dataset)
for image_path, output in zip(image_paths, outputs):
    for cat in ['walkability', 'bikeability', 'pleasantness', 'greenness', 'safety']:
        print(f'{image_path} {cat} score = {output['results'][cat]}')
```

You may also use a simpler API if you wish to forgo the full-blown Dataset-derived class. Simply use:

```python
outputs = pipeline(["image1.jpg", "image2.jpg", "image3.jpg"])
```

## Model Details

The package uses:
- CLIP ViT-H-14-378-quickgelu for feature extraction
- 5 specialized scoring matrices for perception analysis
- Weighted score computation using exponential scaling and softmax normalization

## Requirements

- Python 3.8+
- PyTorch 1.9+
- transformers
- Pillow
- numpy

## Examples

See the [examples](examples/) directory for more detailed usage examples.

## License

GPL-3.0

## Citation

If you use this package in your research, please cite:
```
Danish, M., Labib, SM., Ricker, B., and Helbich, M. A citizen science toolkit to collect human perceptions of urban environments using open street view images. Computers, Environment and Urban Systems. Volume 116, Mar 2025, 102207.
```

This paper is [open access](https://www.sciencedirect.com/science/article/pii/S0198971524001364).

## Support

For issues and feature requests, please use the [GitHub issue tracker](https://github.com/username/svi-percept/issues).
