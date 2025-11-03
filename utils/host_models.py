import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from urllib.parse import urlparse
from unsloth import FastLanguageModel
import uvicorn


print(f"Loading model ...")

load_dotenv()

lora_rank = 8

model_name = os.getenv("OPENAI_MODEL")
if not model_name:
    raise ValueError("Please set the OPENAI_MODEL environment variable.")
model_name_short = model_name.split("/")[-1].lower()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    gpu_memory_utilization = 1.0,
    max_lora_rank = lora_rank,
    random_state = 3407,
)
FastLanguageModel.for_inference(model)

def get_response(messages: list, temperature: float|None = None) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=temperature)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    return response.strip().rstrip("<end_of_turn>").strip()

print("✅ Model loaded.")



app = FastAPI(title="OpenAI-Compatible API", version="1.0")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: int | None = None
    temperature: float | None = None

@app.get("/v1/models")
def list_models():
    return {"data": [{"id": model_name, "object": "model"}]}

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if request.model != model_name:
        return {"error": f"Model '{request.model}' not found"}

    # Generate response
    output = get_response(request.messages, temperature=request.temperature)
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }
        ],
    }


base_url = os.getenv("OPENAI_BASE_URL")
port = urlparse(base_url).port or 8000

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=port)
