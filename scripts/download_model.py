"""Download inference parameters and assets; exclude optimizer/train state."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(os.environ.get('ROBOCASA_ROOT','/root/autodl-tmp/robocasa-repro'))
prefix='pi05_pretrain_human300/multitask_learning/75000'
snapshot_download('robocasa/robocasa365_checkpoints',
    allow_patterns=[f'{prefix}/params/**',f'{prefix}/assets/**',f'{prefix}/_CHECKPOINT_METADATA'],
    local_dir=root/'models/robocasa365_checkpoints',max_workers=8)
print('CHECKPOINT',root/'models/robocasa365_checkpoints'/prefix,flush=True)
