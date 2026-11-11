"""Download inference parameters and assets; exclude optimizer/train state."""
import concurrent.futures,hashlib,json,os,shutil,subprocess
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
root=Path(os.environ.get('ROBOCASA_ROOT','/root/autodl-tmp/robocasa-repro'))
prefix='pi05_pretrain_human300/multitask_learning/75000'
repo='robocasa/robocasa365_checkpoints';dest=root/'models/robocasa365_checkpoints'
if shutil.which('aria2c'):
    api=HfApi();revision=api.model_info(repo).sha
    files=[f for f in api.list_repo_tree(repo,path_in_repo=prefix,revision=revision,recursive=True)
           if hasattr(f,'blob_id') and '/train_state/' not in f.path]
    def get(f):
        path=dest/f.path;path.parent.mkdir(parents=True,exist_ok=True)
        digest=f.lfs.sha256 if f.lfs else None
        if path.exists() and path.stat().st_size==f.size:return
        # Continue the bytes already written by huggingface_hub.
        if not path.exists() and digest:
            cache=dest/'.cache/huggingface/download'/Path(f.path).parent
            matches=list(cache.glob(f'*.{digest}.incomplete')) if cache.exists() else []
            if matches:matches[0].rename(path)
        command=['aria2c','--continue=true','--auto-file-renaming=false','--allow-overwrite=true',
                 '--max-connection-per-server=12','--split=12','--min-split-size=16M',
                 '--file-allocation=none','--max-tries=10','--retry-wait=3',
                 '--console-log-level=warn','--summary-interval=30',
                 '--dir',str(path.parent),'--out',path.name]
        if digest:command.append('--checksum=sha-256='+digest)
        command.append(f'https://huggingface.co/{repo}/resolve/{revision}/{f.path}?download=true')
        subprocess.run(command,check=True)
        if path.stat().st_size!=f.size:raise ValueError('Incorrect checkpoint file size')
        print('MODEL_READY',f.path,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(get,files))
    (dest/'revision.json').write_text(json.dumps({'repo':repo,'revision':revision}))
else:
    snapshot_download(repo,
        allow_patterns=[f'{prefix}/params/**',f'{prefix}/assets/**',f'{prefix}/_CHECKPOINT_METADATA'],
        local_dir=dest,max_workers=8)
print('CHECKPOINT',root/'models/robocasa365_checkpoints'/prefix,flush=True)
