import requests

url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
filename = "llama-2-7b-chat.Q4_K_M.gguf"

print(f"Downloading {filename} (~4GB)... This may take a while.")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
print("Download complete!")