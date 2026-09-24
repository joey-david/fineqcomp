import torch, tempfile
from pathlib import Path
from fineqcomp.generalisation import spectral_slice
from fineqcomp.codec import encode_tensor_map, decode_adapter_tensor_map

raw = torch.load('/tmp/permuted_r16.pt', map_location='cpu', weights_only=True)
band = spectral_slice(raw, 0, 1)
names = [n for n in band if '.lora_A.' in n]
print(f"{len(names)} adapter sites; using all of them")

def stats(dec):
    """<v,u>, ||v||^2, ||u||^2 accumulated over sites without concatenating."""
    dot = n_v = n_u = 0.0
    for a in names:
        b = a.replace('.lora_A.', '.lora_B.')
        u = (band[b].float() @ band[a].float())
        v = (dec[b].float() @ dec[a].float())
        dot += (u*v).sum().item(); n_u += (u*u).sum().item(); n_v += (v*v).sum().item()
    return dot, n_v**0.5, n_u**0.5

_, _, fp16_norm = stats(band)
print(f"rank-1 fp16 band norm: {fp16_norm:.4f}\n")
print(f"{'code':10}{'norm':>10}{'ratio':>8}{'cosine':>9}{'resid_after_best_scale':>24}")
for bits, blend, label in [(1,0.0,"binary"),(1,0.5,"mixed1p5"),(2,0.0,"two_bit"),(4,0.0,"four_bit"),(8,0.0,"eight_bit")]:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)/"a.fqcb"
        encode_tensor_map(band, p, bits=bits, blend=blend,
                          metadata={"singular_start":0,"singular_stop":1})
        _, dec = decode_adapter_tensor_map(p)
    dot, nv, nu = stats(dec)
    cos = dot/(nv*nu)
    alpha = dot/(nu*nu)
    resid = max(nv*nv - 2*alpha*dot + alpha*alpha*nu*nu, 0.0)**0.5 / nv
    print(f"{label:10}{nv:10.4f}{nv/nu:8.3f}{cos:9.4f}{resid:24.4f}")
