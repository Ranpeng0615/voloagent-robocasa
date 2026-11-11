"""Apply the inference-only patch to the pinned upstream checkout."""
import os,subprocess
from pathlib import Path
root=Path(os.environ['ROBOCASA_ROOT']);code=Path(__file__).resolve().parents[1]
for patch in sorted((code/'patches').glob('*.patch')):
    repo = 'openpi' if patch.name.startswith('openpi-') else 'GraspGen' if patch.name.startswith('graspgen-') else 'VoLoAgent'
    cwd=root/'repos'/repo
    check=subprocess.run(['git','apply','--check',str(patch)],cwd=cwd,capture_output=True)
    if check.returncode==0:
        subprocess.run(['git','apply',str(patch)],cwd=cwd,check=True)
    else:
        subprocess.run(['git','apply','--reverse','--check',str(patch)],cwd=cwd,check=True)
    print('PATCH_READY',patch.name,flush=True)
