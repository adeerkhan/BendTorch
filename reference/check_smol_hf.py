"""A one-time check that reference/smol_ref.py is SmolLM2-135M: its loss on the first batch against transformers'.

    <llama.bend venv>/bin/python reference/check_smol_hf.py <hf dir>

Both run in float32 on the same batch (data/smol135/batches.bin, step 0): the twin from data/smol135/init.bin, and
LlamaForCausalLM from the checkpoint in <hf dir> (bfloat16 widened to float32, eager attention). Prints both losses, the
difference, and the largest difference between their logits. Needs transformers, so it runs in llama.bend's venv.
"""

import sys

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM

import smol_ref as R


def main():
    torch.set_num_threads(16)
    x, y = R.batch(0)
    with torch.no_grad():
        ours = R.logits(R.load(), x)
        model = LlamaForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32, attn_implementation="eager").eval()
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()  # tied
        theirs = model(input_ids=x.reshape(R.B, R.T)).logits.reshape(R.B * R.T, R.V)
    a, b = F.cross_entropy(ours, y).item(), F.cross_entropy(theirs, y).item()
    print(f"twin loss {a!r}\ntransformers loss {b!r}\ndifference {a - b:.3e} ({abs(a - b) / abs(b):.3e} relative)")
    print(f"logits: largest difference {(ours - theirs).abs().max().item():.3e}, largest logit "
          f"{theirs.abs().max().item():.3e}")


if __name__ == "__main__":
    main()
