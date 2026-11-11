"""Download official inference weights; leave optimizer/train state out."""
import json,os
from pathlib import Path
from huggingface_hub import HfApi
from ranged_download import download_files
root=Path(os.environ.get('ROBOCASA_ROOT','/root/autodl-tmp/robocasa-repro'))
repo='robocasa/robocasa365_checkpoints'
download_base=os.environ.get('REPRO_HF_DOWNLOAD_BASE_URL','https://huggingface.co').rstrip('/')
prefix='pi05_pretrain_human300/multitask_learning/75000'
dest=root/'models/robocasa365_checkpoints'
api=HfApi();revision=api.model_info(repo).sha;files=[]
for f in api.list_repo_tree(repo,path_in_repo=prefix,revision=revision,recursive=True):
    if not hasattr(f,'blob_id') or '/train_state/' in f.path:continue
    path=dest/f.path;digest=f.lfs.sha256 if f.lfs else None
    # Continue an interrupted huggingface_hub download if present.
    if not path.exists() and digest:
        cache=dest/'.cache/huggingface/download'/Path(f.path).parent
        matches=list(cache.glob(f'*.{digest}.incomplete')) if cache.exists() else []
        if matches:path.parent.mkdir(parents=True,exist_ok=True);matches[0].rename(path)
    files.append({'path':str(path),'size':f.size,'sha256':digest,
                  'url':f'{download_base}/{repo}/resolve/{revision}/{f.path}?download=true'})
print('INFERENCE_GB',sum(f['size'] for f in files)/1e9,flush=True)
download_files(files)
(dest/'revision.json').write_text(json.dumps({'repo':repo,'revision':revision}))
print('CHECKPOINT_READY',dest/prefix,flush=True)
