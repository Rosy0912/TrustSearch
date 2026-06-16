#!/bin/bash
# Download all models + data for Search-R1 experiments.
# Run as: bash download_all.sh
# Requires: huggingface-hub, datasets (pip install huggingface-hub datasets)
set -euo pipefail

MODEL_DIR=/home/jmzhang/models
DATA_DIR=/home/jmzhang/self-evolution/Search-R1/data/retriever_data

echo "================================================================"
echo "[download] Starting all downloads  $(date)"
echo "================================================================"

# ---- 1. Qwen2.5-7B (base) ----
echo "[1/5] Downloading Qwen2.5-7B (base)..."
if [ -f "$MODEL_DIR/Qwen2.5-7B/config.json" ]; then
    echo "  Already exists, skip."
else
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-7B',
    local_dir='$MODEL_DIR/Qwen2.5-7B',
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('[1/5] Qwen2.5-7B done.')
"
fi

# ---- 2. E5-base-v2 (retriever encoder) ----
echo "[2/5] Downloading intfloat/e5-base-v2..."
if [ -f "$MODEL_DIR/e5-base-v2/config.json" ]; then
    echo "  Already exists, skip."
else
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='intfloat/e5-base-v2',
    local_dir='$MODEL_DIR/e5-base-v2',
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('[2/5] e5-base-v2 done.')
"
fi

# ---- 3. Wikipedia-18 corpus + E5 index ----
echo "[3/5] Downloading Wiki-18 corpus + E5 index..."
if [ -f "$DATA_DIR/wiki-18.jsonl" ] && [ -f "$DATA_DIR/e5_Flat.index" ]; then
    echo "  Already exists, skip."
else
    python scripts/download.py --save_path "$DATA_DIR"
    # Merge index
    if [ -f "$DATA_DIR/part_aa" ] && [ -f "$DATA_DIR/part_ab" ]; then
        echo "  Merging index..."
        cat "$DATA_DIR/part_aa" "$DATA_DIR/part_ab" > "$DATA_DIR/e5_Flat.index"
        rm -f "$DATA_DIR/part_aa" "$DATA_DIR/part_ab"
    fi
    # Decompress corpus
    if [ -f "$DATA_DIR/wiki-18.jsonl.gz" ]; then
        echo "  Decompressing wiki-18..."
        gzip -d "$DATA_DIR/wiki-18.jsonl.gz"
    fi
fi

# ---- 4. NQ train/test data (parquet format) ----
echo "[4/5] Processing NQ dataset..."
if [ -f "data/nq_search/train.parquet" ]; then
    echo "  Already exists, skip."
else
    python scripts/data_process/nq_search.py --local_dir data/nq_search
fi

# ---- 5. HotpotQA train/test data ----
echo "[5/5] Processing HotpotQA dataset..."
HOTPOTQA_SCRIPT="scripts/data_process/hotpotqa_search.py"
if [ -f "$HOTPOTQA_SCRIPT" ]; then
    if [ -f "data/hotpotqa_search/train.parquet" ]; then
        echo "  Already exists, skip."
    else
        python "$HOTPOTQA_SCRIPT" --local_dir data/hotpotqa_search 2>/dev/null || echo "  hotpotqa script not available, will use merged script"
    fi
else
    echo "  No hotpotqa_search.py found, will download via qa_search_train_merge.py later"
fi

echo "================================================================"
echo "[download] ALL DONE  $(date)"
echo "  Models -> $MODEL_DIR/Qwen2.5-7B, $MODEL_DIR/e5-base-v2"
echo "  Retriever data -> $DATA_DIR"
echo "  Train data -> data/nq_search/"
echo "================================================================"
