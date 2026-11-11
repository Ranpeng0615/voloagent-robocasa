"""Seed the author's model server, then use its unchanged CLI and endpoints."""
import os,random
import numpy as np
import torch
seed=int(os.environ.get('ROBOCASA_TOOL_SEED','7'))
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from vlm_orchestrator.grasp.server import main
main()
