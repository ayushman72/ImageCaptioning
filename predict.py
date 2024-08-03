import torch
from models import VisionGPT2Model
from transformers import GPT2TokenizerFast
from types import SimpleNamespace
import pathlib
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def download(url:str, filename:str)->pathlib.Path:
    import functools
    import shutil
    import requests
    from tqdm.auto import tqdm
    
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}\n Please download the captioner.pt file manually from the link provided in the README.md file.") 
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path


model_config = SimpleNamespace(
    vocab_size = 50257, # GPT2 vocb size
    embed_dim = 768,    # dim same for both VIT and GPT2
    num_heads = 12,
    seq_len = 1024,
    depth = 12,
    attention_dropout = 0.1,
    residual_dropout = 0.1,
    mlp_ratio = 4,
    mlp_dropout = 0.1,
    emb_dropout = 0.1,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VisionGPT2Model(model_config).to(device)
try:
    sd = torch.load("captioner.pt", map_location=device)
except:
    print("Model not found. Downloading Model ")
    url = "https://drive.usercontent.google.com/download?id=1X51wAI7Bsnrhd2Pa4WUoHIXvvhIcRH7Y&export=download&authuser=0&confirm=t&uuid=ae5c4861-4411-4f81-88cd-66ea30b6fe2b&at=APZUnTWodeDt1upcQVMej2TDcADs%3A1722666079498"
    path = download(url, "captioner.pt")
    sd = torch.load(path, map_location=device)

model.load_state_dict(sd)
model.eval()
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


test_img:str = filedialog.askopenfilename(title = "Select an image",
                                       filetypes = (("jpeg files","*.jpg"),("png files",'*.png'),("all files","*.*")))

tfms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],always_apply=True),
    ToTensorV2()
])


im = Image.open(test_img).convert("RGB")

det = True #generates deterministic results
temp = 1.0 #when det is true, temp has no effect
max_tokens = 50

image = np.array(im)
image:torch.Tensor = tfms(image=image)['image']
image = image.unsqueeze(0).to(device)
seq = torch.ones(1,1).to(device).long()*tokenizer.bos_token_id

caption = model.generate(image, seq, max_tokens, temp, det)
caption = tokenizer.decode(caption.numpy(), skip_special_tokens=True)

plt.imshow(im)
plt.title(f"Predicted : {caption}")
plt.axis('off')
plt.show()


    