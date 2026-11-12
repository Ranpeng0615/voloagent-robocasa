"""Start the isolated model/monitor services, run one episode, then stop them."""
import json,os,signal,socket,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['ROBOCASA_ROOT']);code=Path(__file__).resolve().parents[1]
duplex='--duplex' in sys.argv
task_args=[arg for arg in sys.argv[1:] if arg!='--duplex']
run=root/'runs'/time.strftime(('duplex_' if duplex else 'robocasa_')+'%Y%m%d_%H%M%S');run.mkdir(parents=True)
env={**os.environ,'RUN_DIR':str(run),'PYTHONPATH':str(code/'scripts')}
services=[]
def ready(port):
    with socket.socket() as s:s.settimeout(.2);return s.connect_ex(('127.0.0.1',port))==0
def launch(name,port,command,cwd=None):
    if ready(port):raise RuntimeError(f'Port {port} is already in use')
    log=(run/f'{name}.log').open('wb')
    p=subprocess.Popen(command,env=env,cwd=cwd or code,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    services.append((name,p));(run/'services.json').write_text(json.dumps([{'name':n,'pid':v.pid} for n,v in services]))
    deadline=time.monotonic()+600
    while not ready(port):
        if p.poll() is not None:raise RuntimeError(f'{name} exited; inspect {name}.log')
        if time.monotonic()>deadline:raise TimeoutError(f'{name} startup timeout')
        time.sleep(.5)
    print(name,'READY',flush=True)
try:
    launch('codex',8111,[str(root/'envs/sim/bin/python'),'-m','uvicorn','codex_bridge:app','--host','127.0.0.1','--port','8111'])
    import urllib.request
    with urllib.request.urlopen('http://127.0.0.1:8111/health',timeout=8) as response:
        settings=json.load(response)
    (run/'vlm_config.json').write_text(json.dumps(settings,indent=2))
    launch('policy',8000,['bash',str(code/'scripts/start_policy.sh')])
    with (run/'policy_warmup.json').open('w') as f:
        subprocess.run([str(root/'envs/sim/bin/python'),str(code/'scripts/policy_warmup.py')],env=env,stdout=f,check=True,timeout=240)
    if duplex:
        launch('tools',8003,['bash',str(code/'scripts/start_tools.sh')])
        with (run/'tools_warmup.json').open('w') as f:
            subprocess.run([str(root/'envs/sim/bin/python'),str(code/'scripts/tools_warmup.py')],env=env,stdout=f,check=True,timeout=360)
    else:
        launch('orchestrator',8001,['bash',str(code/'scripts/start_monitor.sh')])
    print('RUN',run,flush=True)
    with (run/'task.log').open('wb') as f:
        entry='run_duplex.py' if duplex else 'run_one.py'
        task=subprocess.Popen([str(root/'envs/sim/bin/python'),str(code/'scripts'/entry),*task_args],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        services.append(('task',task));watch=time.monotonic()+600
        while task.poll() is None:
            state=run/'wall_state.json'
            if state.exists():
                try:s=json.loads(state.read_text())
                except json.JSONDecodeError:s={}
                if s.get('phase')=='running' and time.monotonic()>s['deadline_monotonic']+3:
                    os.killpg(task.pid,signal.SIGKILL);break
            elif time.monotonic()>watch:
                os.killpg(task.pid,signal.SIGKILL);break
            time.sleep(.1)
        task.wait()
    if (run/'result.json').exists():print((run/'result.json').read_text(),flush=True)
    if task.returncode:raise RuntimeError('Episode process failed; inspect result.json and task.log')
finally:
    for name,p in reversed(services):
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
