"""Check out the pinned public upstream repositories."""
import json, os, subprocess
from pathlib import Path
root=Path(os.environ.get('ROBOCASA_ROOT','/root/autodl-tmp/robocasa-repro'))
code=Path(__file__).resolve().parents[1]
for name,spec in json.loads((code/'configs/upstreams.json').read_text()).items():
    dest=root/'repos'/name
    if not (dest/'.git').exists():
        subprocess.run(['git','clone','--no-checkout',spec['url'],str(dest)],check=True)
    current=subprocess.run(['git','rev-parse','HEAD'],cwd=dest,capture_output=True,text=True)
    if current.returncode or current.stdout.strip()!=spec['commit'] or not (dest/'.git/index').exists():
        present=subprocess.run(['git','cat-file','-e',spec['commit']+'^{commit}'],cwd=dest,capture_output=True)
        if present.returncode:
            subprocess.run(['git','fetch','origin',spec['commit']],cwd=dest,check=True)
        subprocess.run(['git','checkout','--detach',spec['commit']],cwd=dest,check=True)
    print(name,spec['commit'],flush=True)
