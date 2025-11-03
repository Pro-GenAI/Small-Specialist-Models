import os
import concurrent.futures
import json
import hashlib

from deepeval.models.base_model import DeepEvalBaseLLM
import openai
from dotenv import load_dotenv
load_dotenv()

model_name = os.getenv("OPENAI_MODEL") or ""
if not model_name:
    raise ValueError("OPENAI_MODEL environment variable not set")

client = openai.OpenAI()


_sample_cache: dict[str, str | list[str]] = {}
_sample_cache_path = os.path.join(os.path.dirname(__file__), "sample_cache.json")
try:
    with open(_sample_cache_path, encoding="utf-8") as f:
        _sample_cache = json.load(f)
except FileNotFoundError:
    _sample_cache = {}
except Exception as e:
    print(f"Warning: Failed to load sample cache: {e}")
    raise e

def save_cache():
    try:
        tmp_path = _sample_cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_sample_cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, _sample_cache_path)
    except Exception:
        # If saving fails, continue without raising to avoid breaking generation
        pass


def generate_response(message: list[dict[str, str]] | str,
                      temperature: float | None = None) -> str:
    messages = message if isinstance(message, list) else [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model=model_name, messages=messages, # type: ignore
        temperature=temperature,
    )
    response = response.choices[0].message.content
    if not response:
        raise Exception("Empty response from the bot")
    return response.strip()


counter = 0

def get_response(prompt: str, temperature: float|None = None, cache: bool = True) -> str:
    global counter
    counter += 1

    # Find cached response if available
    key_input = f"{model_name}||{prompt}||{temperature}"
    key = hashlib.sha256(key_input.encode("utf-8")).hexdigest()
    if cache and key in _sample_cache and _sample_cache[key]:
        print(f"Using cached response {counter}...")
        return _sample_cache[key][0]

    print(f"Generating response {counter}...")
    response = generate_response(prompt, temperature=temperature)

    _sample_cache[key] = response
    save_cache()
    return response

def batch_generate(prompts: list[str]) -> list[str]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        responses = list(executor.map(generate_response, prompts))
    return responses


class CustomModel(DeepEvalBaseLLM):
    def __init__(self, apply_function: callable | None = None): # type: ignore
        self.apply_function = apply_function

    def load_model(self):  # type: ignore
        return True

    def generate(self, prompt: str, temperature: float|None = None, cache: bool = True) -> str:
        return get_response(prompt, temperature=temperature, cache=cache)

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return get_response(prompt)

    def batch_generate(self, prompts: list[str]) -> list[str]:
        return batch_generate(prompts)

    def generate_samples(self, prompt: str, n: int, temperature: float) -> list[str]:
        # Includes caching of raw responses to avoid redundant generation

        key_input = f"{model_name}||{prompt}||{temperature}"
        key = hashlib.sha256(key_input.encode("utf-8")).hexdigest()

        responses: list[str] = list(_sample_cache.get(key, []))
        while len(responses) < n:
            response = self.generate(prompt, temperature=temperature, cache=False)
            responses.append(response)
            _sample_cache[key] = responses
            save_cache()

        if self.apply_function:
            return [self.apply_function(r) for r in responses[:n]]
        return responses[:n]

    def get_model_name(self):  # type: ignore
        return model_name


if __name__ == "__main__":
    response = get_response("Hi")
    print(response)
