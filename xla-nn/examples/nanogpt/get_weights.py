# Retrieve the GPT2 weights from HuggingFace.

import numpy as np
import transformers
from safetensors.numpy import save_file

model_name = "gpt2"
model = transformers.GPT2LMHeadModel.from_pretrained(model_name)

# The HF GPT2 linear layers use Conv1D which stores its weight as
# [in_features, out_features]; xla_nn::Linear expects the PyTorch
# [out_features, in_features] layout so these get transposed.
TRANSPOSED = [
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
]

numpy_arrays = {}
for k, v in model.state_dict().items():
    if k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"):
        continue
    v = v.numpy()
    if any(k.endswith(t) for t in TRANSPOSED):
        v = np.ascontiguousarray(np.transpose(v))
    print(k, v.shape, v.dtype)
    numpy_arrays[k] = v
save_file(numpy_arrays, f"{model_name}.safetensors")
