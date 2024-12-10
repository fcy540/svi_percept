from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from .pipeline import SVIPerceptPipeline

# Register the pipeline
PIPELINE_REGISTRY.register_pipeline(
    "svi-percept",  # Task identifier
    pipeline_class=SVIPerceptPipeline,
    pt_model=None  # We don't need a default model since we use CLIP directly
)
