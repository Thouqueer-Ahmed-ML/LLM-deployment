"""
Module providing a Ray Actor which helps to deploy an Embedding model.
Current code:
    Accepts a simple string in the form of HTTP JSON payload
    Returns a JSON list of decimals
"""
import typing
import torch
import ray

from starlette.requests import Request
from transformers import AutoTokenizer, AutoModel


# This model gets downloaded from hugging-face and gets stored in our disk for future use
# Model download == downloading it's structure and the weights i.e., parameters
MODEL_ID = "BAAI/bge-large-zh"


# Ray allows you to specify the maximum number of CPUs/GPUs you want this model to reserve
@ray.serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2})
class EmbeddingDeployment:
    """
    Class representing a Ray actor.
    It consists of an asynchronous __call__ method which gets triggered on API requests
    It generates embeddings for a given input-string/prompt
    """

    def __init__(self):

        self.model = AutoModel.from_pretrained(MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def generate(self, prompt: str) -> torch.Tensor:
        """
        This function generates and returns the embeddings as a PyTorch tensor
        """

        # Use tokenizer to convert prompt into tokens
        encoded_input = self.tokenizer([prompt], padding=True, truncation=True, return_tensors='pt')

        # Disable auto-gradient tracker in PyTorch as it is not needed
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]

        # Normalise to single dimension
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    async def __call__(self, http_request: Request) -> typing.List[float]:

        json_request: str = await http_request.json()
        return self.generate(json_request).tolist()[0]


# Create a Ray actor
deployment = EmbeddingDeployment.bind()
