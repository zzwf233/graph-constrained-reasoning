import time
import os
from openai import OpenAI
from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken
dotenv.load_dotenv()

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

OPENAI_MODEL = ['gpt-4', 'gpt-3.5-turbo']
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
DEFAULT_API_BASE = "https://api.siliconflow.cn/v1"
MODEL_TO_ENCODING = {
    DEFAULT_MODEL_NAME.lower(): "cl100k_base",
    "deepseek-ai/deepseek-v3.2": "cl100k_base",
}

def get_token_limit(model='gpt-4'):
    """Returns the token limitation of provided model"""
    if model in ['gpt-4', 'gpt-4-0613']:
        num_tokens_limit = 8192
    elif model in ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']:
        num_tokens_limit = 128000
    elif model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613']:
        num_tokens_limit = 16384
    elif model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'text-davinci-003', 'text-davinci-002']:
        num_tokens_limit = 4096
    elif model.lower() in [DEFAULT_MODEL_NAME.lower(), "qwen/qwen2.5-vl-72b-instruct"]:
        num_tokens_limit = 128000
    elif model.lower() in ["deepseek-ai/deepseek-v3.2"]:
        num_tokens_limit = 128000
    else:
        # default for OpenAI-compatible long-context chat models
        num_tokens_limit = 128000
    return num_tokens_limit

PROMPT = """{instruction}

{input}"""

class ChatGPT(BaseLanguageModel):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--retry', type=int, help="retry time", default=5)
        parser.add_argument('--model_path', type=str, default='None')
        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
        parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
           
    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = args.model_name if args.model_name else DEFAULT_MODEL_NAME
        self.maximun_token = get_token_limit(self.model_name)
        
    def token_len(self, text):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding_name = MODEL_TO_ENCODING.get(self.model_name.lower())
            if encoding_name:
                encoding = tiktoken.get_encoding(encoding_name)
            else:
                encoding = tiktoken.encoding_for_model(self.model_name)
            return len(encoding.encode(text))
        except Exception:
            try:
                # For OpenAI-compatible custom model IDs, fall back to cl100k_base.
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                # Final fallback when tokenizer files are unavailable in restricted networks.
                return max(1, len(text) // 4)
    
    def prepare_for_inference(self, model_kwargs={}):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in your environment or .env."
            )
        client_kwargs = {"api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or DEFAULT_API_BASE
        client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
    
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        return query
    
    def generate_sentence(self, llm_input):
        cur_retry = 0
        num_retry = self.retry
        # Chekc if the input is too long
        input_length = self.token_len(llm_input)
        if input_length > self.maximun_token:
            print(f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens.")
            llm_input = llm_input[:self.maximun_token]
        query = [{"role": "user", "content": llm_input}]
        while cur_retry <= num_retry:
            try:
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = query,
                    timeout=60,
                    temperature=0.0
                    )
                result = response.choices[0].message.content.strip() # type: ignore
                return result
            except Exception as e:
                print("Message: ", llm_input)
                print("Number of token: ", self.token_len(llm_input))
                print(e)
                err_msg = str(e).lower()
                if "429" in err_msg or "rate limit" in err_msg or "tpm limit reached" in err_msg:
                    # Exponential backoff with jitter for API rate limits.
                    import random as _random
                    sleep_time = min(300, 15 * (2 ** cur_retry)) + _random.uniform(0, 3)
                else:
                    sleep_time = 30
                print(f"Retry in {sleep_time:.1f}s ...")
                time.sleep(sleep_time)
                cur_retry += 1
                continue
        return None
