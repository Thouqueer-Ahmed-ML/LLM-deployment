"""
Module providing a Ray Actor which helps to deploy a Re-Ranking model.
Current code:
    Accepts a JSON payload like
    {
      "query": "Tell me about India",
      "documents": [
        "This is about India",
        "This is about America",
        "This is about Japan"
      ],
      "top_n": 3
    }
    Returns a JSON response like
    {
      "results": [
        {
          "document": {
            "text": "This is about India"
          },
          "index": 0,
          "relevance_score": 0.12262824
        },
        {
          "document": {
            "text": "This is about America"
          },
          "index": 1,
          "relevance_score": 0.012673736
        }
      ]
    }
"""
import typing

import torch
import ray
from starlette.requests import Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_ID = "BAAI/bge-reranker-large"


# Ray allows you to specify the maximum number of CPUs/GPUs you want this model to reserve
@ray.serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
class ReRankerDeployment:
    """
    Class representing a Ray actor.
    It consists of an asynchronous __call__ method which gets triggered on API requests
    It assigns score to the items in list of documents and returns the top_n scoring documents
    """

    def __init__(self):

        torch.cuda.set_device(0)

        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

        self.model = self.model.to(device='cuda:0')

        self.model = torch.nn.DataParallel(self.model, device_ids=[0])

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def generate(self, query: str, documents: typing.List[str], top_n: int) -> dict:
        """
        This function scores the documents and returns the top_n scoring documents
        """

        # API payload is different from model's expected input. Transform it into [[query, doc1], [query, doc2] ...]
        model_input = list()
        for document in documents:
            model_input.append([query, document])

        with torch.no_grad():  # Disable auto-gradient tracker in PyTorch as it is not needed
            inputs = self.tokenizer(model_input, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores: torch.Tensor = self.model(**inputs, return_dict=True).logits.view(-1, ).float()  # Score computation

        scores: list = scores.tolist()

        docs_with_scores = [(documents[indx], scores[indx], indx) for indx in range(len(documents))]
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        docs_with_scores = docs_with_scores[:top_n]

        print(docs_with_scores)

        return {
            "results": [
                {
                    "document": {
                        "text": docs_with_scores[indx][0]
                    },
                    "relevance_score": docs_with_scores[indx][1],
                    "index": docs_with_scores[indx][2]
                }
                for indx in range(len(docs_with_scores))
            ]
        }

    async def __call__(self, http_request: Request):

        json_request: dict = await http_request.json()
        return self.generate(json_request["query"], json_request["documents"], json_request["top_n"])


# Create a Ray actor
deployment = ReRankerDeployment.bind()
