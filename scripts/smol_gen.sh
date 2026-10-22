#!/usr/bin/env bash
# SmolLM2-135M writing text in BendTorch (demo/smol.bend's gen and draw modes), checked against its PyTorch twin.
# Encodes the prompt with the model's tokenizer, continues it greedily in Bend and in reference/smol_ref.py's gen mode
# (the same window: the tokens at its start, token 0 after them), requires the same tokens, then draws the same number
# of tokens at temperature 0.8 with a fixed seed (not checked: greedy decoding loops on this model) and prints both
# texts. Both are run again with the key/value cache (genkv, drawkv), which must give the same tokens. The weights are
# init.bin (the pinned checkpoint) or a file written by `finetune <k> <e> <out>`.
#
#   scripts/smol_gen.sh [weights] [tokens] [seed] [prompt]   # LLAMA_BEND=<path to llama.bend> if not ../llama.bend
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-data/smol135/init.bin}
N=${2:-60}
SEED=${3:-7}
PROMPT=${4:-$'ROMEO:\n'}
LB=$(realpath "${LLAMA_BEND:-../llama.bend}")
HF=$LB/data/smollm2-135m/hf
TPY=$LB/reference/.venv/bin/python
PY=reference/.venv/bin/python
[ -f "$HF/tokenizer.json" ] && [ -x "$TPY" ] || { echo "no SmolLM2-135M tokenizer or venv in $LB (set LLAMA_BEND)"; exit 1; }
[ -x .scratch/smol ] && [ .scratch/smol -nt demo/smol.bend ] || bend demo/smol.bend -o .scratch/smol
decode() { "$TPY" reference/smol_ref.py --decode "$HF" "$1" 2>/dev/null; }
IDS=$("$TPY" reference/smol_ref.py --encode "$HF" "$PROMPT" 2>/dev/null)
echo "weights $W, prompt ids $IDS"
B=$(.scratch/smol --threads 16 gen "$N" "$W" "$IDS")
R=$(THREADS=16 "$PY" reference/smol_ref.py gen "$N" "$W" "$IDS")
echo "$B" | grep gen_ms
BI=$(echo "$B" | sed -n 's/^ids //p')
RI=$(echo "$R" | sed -n 's/^ids //p')
if [ "$BI" = "$RI" ]; then
  echo "greedy: the same $(( $(echo "$BI" | wc -w) - $(echo "$IDS" | tr ',' ' ' | wc -w) )) tokens after the prompt as the twin"
else
  echo "greedy: Bend and the twin differ"; echo "bend  $BI"; echo "torch $RI"; exit 1
fi
K=$(.scratch/smol --threads 16 genkv "$N" "$W" "$IDS")
echo "$K" | grep gen_ms | sed 's/^/cached /'
KI=$(echo "$K" | sed -n 's/^ids //p')
[ "$KI" = "$BI" ] && echo "greedy: the same tokens with the key/value cache" || { echo "greedy: the cache gives other tokens"; echo "cached $KI"; exit 1; }
echo "--- greedy"; decode "$BI"
D=$(.scratch/smol --threads 16 draw "$N" "$SEED" "$W" "$IDS")
echo "--- drawn (temperature 0.8, seed $SEED)"; decode "$(echo "$D" | sed -n 's/^ids //p')"
echo "$D" | sed -n 's/^ids /draw_ids /p'
DK=$(.scratch/smol --threads 16 drawkv "$N" "$SEED" "$W" "$IDS")
[ "$(echo "$DK" | sed -n 's/^ids //p')" = "$(echo "$D" | sed -n 's/^ids //p')" ] && echo "drawn: the same tokens with the key/value cache" || { echo "drawn: the cache gives other tokens"; echo "$DK"; exit 1; }
