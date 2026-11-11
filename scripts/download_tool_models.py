"""Fetch official, pinned tool checkpoints or verify existing local copies."""
import hashlib,json,os,shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
root=Path(os.environ['ROBOCASA_ROOT'])
code=Path(__file__).resolve().parents[1]
specs=json.loads((code/'configs/tool-models.json').read_text())
for name,spec in specs.items():
    for filename,info in spec['files'].items():
        dest=root/'models'/info['destination']
        if not dest.is_file():
            # SAM3 is gated. Use the operator's licensed HF access token through
            # the standard HF cache/environment; never store it in the repo.
            source=hf_hub_download(repo_id=spec['repo'],filename=filename,revision=spec['revision'])
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
        if 'sha256' in info:
            digest=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
            if digest!=info['sha256']: raise ValueError('Checkpoint digest mismatch: '+str(dest))
        print('VERIFIED',name,filename,flush=True)
