"""
Module providing a Ray Actor which helps to deploy an LLM model.
Current code:
    Accepts a simple string in the form of HTTP JSON payload
    Returns a JSON object of the form {"text": "Answer to your prompt"}
"""
import typing
import ray

from starlette.requests import Request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as tx_pipeline
from langchain import PromptTemplate, LLMChain, HuggingFacePipeline


# This model gets downloaded from hugging-face and gets stored in our disk for future use
# Model download == downloading it's structure and the weights i.e., parameters
MODEL_ID = "NousResearch/Llama-2-13b-chat-hf"


# Ray allows you to specify the maximum number of CPUs/GPUs you want this model to reserve
@ray.serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 6})
class LLMDeployment:
    """
    Class representing a Ray actor.
    It consists of an asynchronous __call__ method which gets triggered on API requests
    It generates a textual response for a given input-string/prompt
    It provides a Question & Answer kind of interface rather than a chat interface but it can easily modified to support conversations
    """

    def __init__(self):

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",  # Equally distributes the Llama model on all the GPUs present in your system. It uses Hugging-Face's accelerate library
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        '''
        Create a hugging-face pipeline which
            1. tokenizes the given prompt to generate tokens
            2. sends to tokens to model
            3. converts output tokens into words/phrases
        '''
        pipeline = tx_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            use_cache=True,
            max_length=4096,  # Llama has a token limit of 4096 tokens. Set max_length to this value
            do_sample=False,  # This is equivalent to setting temperatue=0 in OpenAI APIs
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        # Use LangChain wrapper to create an easy interface around the hugging-face pipeline
        llm = HuggingFacePipeline(pipeline=pipeline)

        # This template can be modified to add system prompts etc.
        template = "{question}"

        prompt2 = PromptTemplate(
            template=template,
            input_variables=["question"]
        )

        self.llm_chain = LLMChain(prompt=prompt2, llm=llm)

    def generate(self, prompt: str) -> typing.Dict[str, str]:
        """
        This function generates and returns the embeddings as a python dictionary of the form
        {"text": "Answer to your prompt"}
        """

        return self.llm_chain(prompt.strip())

    async def __call__(self, http_request: Request) -> typing.Dict[str, str]:

        json_request: str = await http_request.json()
        return self.generate(json_request)


deployment = LLMDeployment.bind()
