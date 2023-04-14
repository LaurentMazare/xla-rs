# Retrieve the GPT2 weights from HuggingFace.

import numpy as np
import transformers

model_name = "gpt2"
model = transformers.GPT2LMHeadModel.from_pretrained(model_name)

TRANSPOSED = set([
    "lm_head.weight"
])

numpy_arrays = {}
for k, v in model.state_dict().items():
    if k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"):
        continue
    v = v.numpy()
    if k in TRANSPOSED:
        v = np.ascontiguousarray(np.transpose(v))
    print(k, v.shape, v.dtype)
    numpy_arrays[k] = v
np.savez(f"{model_name}.npz", **numpy_arrays)
