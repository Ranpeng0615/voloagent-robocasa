"""Compile the inference graph before timing an environment episode."""
import json,time
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
image=np.zeros((224,224,3),dtype=np.uint8)
obs={'observation/image':image,'observation/right_image':image,
     'observation/wrist_image':image,'observation/state':np.zeros(16,dtype=np.float32),
     'prompt':'pick up the object'}
t=time.monotonic();result=WebsocketClientPolicy('127.0.0.1',8000).infer(obs)
actions=np.asarray(result['actions'])
assert actions.ndim==2 and actions.shape[1]==12 and np.isfinite(actions).all()
print(json.dumps({'warmup_s':time.monotonic()-t,'action_shape':list(actions.shape),'finite':True}))
