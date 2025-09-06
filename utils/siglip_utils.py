# utils/siglip_utils.py
import torch, PIL.Image as Image, numpy as np
from transformers import SiglipProcessor, SiglipModel
# --- add this mapping near the top ---
SIGLIP_ALIASES = {
    "ViT-B-32": "google/siglip-so400m-patch14-384",
    "ViT-L-14": "google/siglip-large-patch16-384",
    "ViT-L-14-336": "google/siglip-large-patch16-384",
    "SO400M/14-384": "google/siglip-so400m-patch14-384",
    "Large/16-384": "google/siglip-large-patch16-384",
    "Base/16-384": "google/siglip-base-patch16-384",
    "Base/16-512": "google/siglip-base-patch16-512",
}

def _resolve_id(name: str) -> str:
    if name in SIGLIP_ALIASES:
        return SIGLIP_ALIASES[name]
    # If it's not a full repo id, fall back to a strong default
    if "/" not in name:
        return "google/siglip-so400m-patch14-384"
    return name
# --- end mapping ---

def _load(model_name="google/siglip-so400m-patch14-384"):
    model_name = _resolve_id(model_name)   # <-- use the resolver here
    model = SiglipModel.from_pretrained(model_name).to(_device())
    proc  = SiglipProcessor.from_pretrained(model_name)
    model.eval()
    return model, proc

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _load(model_name="google/siglip-so400m-patch14-384"):
    model = SiglipModel.from_pretrained(model_name).to(_device())
    proc  = SiglipProcessor.from_pretrained(model_name)
    model.eval()
    return model, proc

@torch.no_grad()
def compute_image_embeddings(paths, model_name="google/siglip-so400m-patch14-384",
                             pretrained=None, batch_size=64):
    model, proc = _load(model_name)
    vecs = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
        inputs = proc(images=imgs, return_tensors="pt").to(_device())
        out = model(**inputs)
        v = out.image_embeds  # (B, D)
        v = torch.nn.functional.normalize(v, dim=1)
        vecs.append(v.cpu().numpy())
    return np.concatenate(vecs, 0).astype("float32")

@torch.no_grad()
def compute_text_embeddings(captions_list, model_name="google/siglip-so400m-patch14-384",
                            pretrained=None, aggregate="average"):
    """
    captions_list: List[List[str]] (your app format). We return one vector per item.
    If multiple captions, we either 'average' or take 'first'.
    """
    model, proc = _load(model_name)
    out_vecs = []
    for caps in captions_list:
        caps = caps if isinstance(caps, (list, tuple)) and len(caps) else [""]
        inputs = proc(text=caps, padding=True, return_tensors="pt").to(_device())
        out = model(**inputs)
        t = torch.nn.functional.normalize(out.text_embeds, dim=1)  # (k, D)
        if len(caps) > 1:
            if aggregate == "first":
                t = t[0:1]
            else:
                t = t.mean(0, keepdim=True)
        out_vecs.append(t.cpu().numpy())
    return np.concatenate(out_vecs, 0).astype("float32")

def cosine_similarity(A, B):
    import numpy as np
    return (A @ B.T)
