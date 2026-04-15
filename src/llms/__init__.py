from .chatgpt import ChatGPT
from .base_language_model import BaseLanguageModel

registed_language_models = {
    'gpt': 'chatgpt',
    'qwen2.5-vl-72b-instruct': 'chatgpt',
    'deepseek-v3.2': 'chatgpt',
    'gcr': 'gcr',
    'others': 'hf',
}


def _load_model_class(model_type: str):
    if model_type == "chatgpt":
        return ChatGPT
    if model_type == "gcr":
        from .graph_constrained_decoding_model import GraphConstrainedDecodingModel
        return GraphConstrainedDecodingModel
    from .base_hf_causal_model import HfCausalModel
    return HfCausalModel

def get_registed_model(model_name) -> BaseLanguageModel:
    for key, value in registed_language_models.items():
        if key in model_name.lower():
            return _load_model_class(value)
    print("Model is not found in the registed_language_models, return HfCausalModel by default")
    return _load_model_class("hf")