from transformers import PreTrainedModel, PretrainedConfig
from sklearn.utils.extmath import softmax
import numpy as np
import torch
from huggingface_hub import hf_hub_download


class SVIPerceptConfig(PretrainedConfig):
    model_type = "svi_percept"

    def __init__(self,
                repo_id: str = "Spatial-Data-Science-and-GEO-AI-Lab/svi_percept",  # Default repo
                **kwargs):
        self.repo_id = repo_id
        self.categories = ['walkability', 'bikeability', 'pleasantness', 'greenness', 'safety']
        self.k = 40
        super().__init__(**kwargs)

class SVIPerceptModel(PreTrainedModel):
    config_class = SVIPerceptConfig

    def __init__(self, config, repo_id=None, categories=None, k=None):
        super().__init__(config)
        self.repo_id = repo_id or config.repo_id
        self.categories = categories or config.categories
        self.k = k or config.k
        # FIXME/future work: run the model in terms of torch rather than numpy
        self.dummy = torch.nn.Linear(1,2) # avoids issues with torch
        # Load npz data from model repository
        npz_path = hf_hub_download(
            repo_id=self.repo_id,
            filename="data.npz",
            repo_type="model"
        )
        self.data = np.load(npz_path)

    def forward(self, features, **kwargs):
        results = np.zeros((features.shape[0], len(self.categories)))
        debug = False
        features = features.numpy()
        k = self.k
        if debug: print('features.shape', features.shape)
        for cat_i, cat in enumerate(self.categories):
            allvecs = self.data[f'{cat}_vecs']
            if debug: print('allvecs.shape', allvecs.shape)
            scores = self.data[f'{cat}_scores']
            if debug: print('scores.shape', scores.shape)
            # Compute cosine similiarity of vec against allvecs
            # (both are already normalized)
            cos_sim_table = features @ allvecs.T
            if debug: print('cos_sim_table.shape', cos_sim_table.shape)
            # Get sorted array indices by similiarity in descending order
            sortinds = np.flip(np.argsort(cos_sim_table, axis=1), axis=1)
            if debug: print('sortinds.shape', sortinds.shape)
            # Get corresponding scores for the sorted vectors
            kscores = scores[sortinds][:,:k]
            if debug: print('kscores.shape', kscores.shape)
            # Get actual sorted similiarity scores
            ksims = cos_sim_table[np.expand_dims(np.arange(sortinds.shape[0]), axis=1), sortinds]
            ksims = ksims[:,:k]
            if debug: print('ksims.shape', ksims.shape)
            # Apply normalization after exponential formula
            ksims = softmax(10**ksims)
            #ksims = ksims / np.sum(ksims)
            # Weighted sum
            kweightedscore = np.sum(kscores * ksims, axis=1)
            results[:, cat_i] = kweightedscore
        return {"results": results}
