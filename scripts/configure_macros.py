"""Set dataset path metadata without downloading training demonstrations."""
import os
from pathlib import Path
root=Path(os.environ['ROBOCASA_ROOT'])
p=root/'repos/robocasa/robocasa/macros_private.py'
p.write_text('DATASET_BASE_PATH = '+repr(str(root/'data/datasets'))+'\n')
(root/'data/datasets').mkdir(parents=True,exist_ok=True)
print('Configured RoboCasa dataset path metadata')
