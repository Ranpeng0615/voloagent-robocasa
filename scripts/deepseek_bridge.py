"""Persistent local adapter for DeepSeek visual monitoring."""
import json,os,time,uuid
from pathlib import Path
import httpx
from fastapi import FastAPI,HTTPException

root=Path(os.environ['ROBOCASA_ROOT'])
key_file=Path(os.environ.get('DEEPSEEK_KEY_FILE',root/'secrets/deepseek.key'))
client=httpx.AsyncClient(base_url='https://api.deepseek.com',timeout=180,
    headers={'Authorization':'Bearer '+key_file.read_text().strip()})
app=FastAPI()
@app.on_event('shutdown')
async def shutdown():await client.aclose()
@app.get('/health')
def health():return {'status':'ok','model':'deepseek-flash','reasoning_effort':'max'}
@app.post('/v1/chat/completions')
async def completion(body:dict):
    payload=dict(body)
    payload.update(model='deepseek-flash',stream=False,reasoning_effort='max',thinking={'type':'enabled'})
    for key in ('max_tokens','max_completion_tokens','stream_options'):payload.pop(key,None)
    folder=Path(os.environ.get('RUN_DIR',root/'runs'))/'deepseek_calls'/str(uuid.uuid4())
    folder.mkdir(parents=True)
    (folder/'request.json').write_text(json.dumps(payload))
    start=time.monotonic()
    try:
        response=await client.post('/chat/completions',json=payload)
        response.raise_for_status();result=response.json()
    except (httpx.HTTPError,ValueError) as exc:
        (folder/'error.json').write_text(json.dumps({'type':type(exc).__name__,'elapsed_s':time.monotonic()-start}))
        raise HTTPException(502,'DeepSeek request failed; inspect server run logs') from None
    (folder/'response.json').write_text(json.dumps({'elapsed_s':time.monotonic()-start,**result}))
    return result
