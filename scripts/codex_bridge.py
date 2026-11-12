"""Local OpenAI-compatible bridge to the user's authenticated Codex CLI."""
import asyncio
import base64
import json
import os
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, HTTPException
import httpx

ROOT = Path(os.environ.get('ROBOCASA_ROOT', Path(__file__).resolve().parents[1]))
MODEL = 'gpt-6.1-sol'
UPSTREAM = os.environ.get('CODEX_BRIDGE_UPSTREAM', '').rstrip('/')
app = FastAPI()
lock = asyncio.Lock()

@app.get('/health')
async def health():
    if UPSTREAM:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            try:
                response=await client.get(UPSTREAM+'/health')
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError:
                raise HTTPException(503, 'Local Codex relay is unavailable') from None
    return {'status':'ok', 'model':MODEL, 'reasoning_effort':'high', 'service_tier':'fast'}

@app.get('/v1/models')
def models():
    return {'object':'list','data':[{'id':MODEL,'object':'model'}]}

@app.post('/v1/chat/completions')
async def completion(body: dict):
    if UPSTREAM:
        folder=Path(os.environ.get('RUN_DIR',ROOT/'runs'))/'codex_calls'/str(uuid.uuid4())
        folder.mkdir(parents=True)
        (folder/'request.json').write_text(json.dumps(body))
        start=time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=190, trust_env=False) as client:
                response=await client.post(UPSTREAM+'/v1/chat/completions',json=body)
                response.raise_for_status()
                result=response.json()
        except (httpx.HTTPError, ValueError) as exc:
            (folder/'error.json').write_text(json.dumps({'type':type(exc).__name__, 'elapsed_s':time.monotonic()-start}))
            raise HTTPException(502, 'Codex relay request failed; inspect relay logs') from None
        (folder/'response.json').write_text(json.dumps({'elapsed_s':time.monotonic()-start, **result}))
        return result
    async with lock:
        ident = str(uuid.uuid4())
        folder = Path(os.environ.get('RUN_DIR',ROOT/'runs')) / 'codex_calls' / ident
        folder.mkdir(parents=True)
        (folder/'request.json').write_text(json.dumps(body))
        prompt = ['You are the visual reasoning backend of a robot simulator. Read the attached images and instructions. Return only the requested answer, normally a JSON object. Do not use tools or inspect files.']
        images = []
        for message in body.get('messages', []):
            prompt.append('\nROLE: '+message.get('role','user'))
            content = message.get('content') or ''
            if isinstance(content,str):
                prompt.append(content)
                continue
            for part in content:
                if part.get('type') == 'text':
                    prompt.append(part['text'])
                elif part.get('type') == 'image_url':
                    url=part['image_url']['url']
                    if not url.startswith('data:image/'):
                        raise HTTPException(400,'Only embedded image inputs are supported')
                    header,data=url.split(',',1)
                    ext='png' if 'png' in header else 'jpg'
                    path=folder/f'image_{len(images)}.{ext}'
                    path.write_bytes(base64.b64decode(data))
                    images.append(path)
                    prompt.append(f'[Attached image {len(images)}]')
        text='\n'.join(prompt)
        (folder/'prompt.txt').write_text(text)
        answer=folder/'answer.txt'
        args=['codex','exec','--ignore-user-config','--model',MODEL,'-c','model_reasoning_effort="high"','-c','service_tier="fast"','--disable','plugins','--disable','remote_plugin','--sandbox','read-only','--skip-git-repo-check','--ephemeral','-C',str(ROOT),'--output-last-message',str(answer)]
        catalog=os.environ.get('CODEX_MODEL_CATALOG')
        if catalog: args += ['-c', 'model_catalog_json='+json.dumps(catalog)]
        for path in images:
            args += ['--image',str(path)]
        args += ['-']
        start=time.monotonic()
        with (folder/'stdout.log').open('wb') as out, (folder/'stderr.log').open('wb') as err:
            proc=await asyncio.create_subprocess_exec(*args,stdin=asyncio.subprocess.PIPE,stdout=out,stderr=err)
            try:
                await asyncio.wait_for(proc.communicate(text.encode()),timeout=180)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise HTTPException(504,'Codex inference exceeded 180 seconds')
        if proc.returncode or not answer.exists():
            raise HTTPException(502,f'Codex failed; inspect {folder}')
        response_text=answer.read_text().strip()
        result={'id':'chatcmpl-'+ident,'object':'chat.completion','created':int(time.time()),'model':MODEL,'choices':[{'index':0,'message':{'role':'assistant','content':response_text},'finish_reason':'stop'}], 'usage':None, 'requested_service_tier':'fast', 'reasoning_effort':'high'}
        (folder/'response.json').write_text(json.dumps({'elapsed_s':time.monotonic()-start,**result}))
        return result
