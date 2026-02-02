import requests

API_URL = "http://localhost:8002/v1/chat/completions"

prompt = "Hello from vllm serve!"
data = {
            "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                }

resp = requests.post(API_URL, json=data)
print(resp.json())

