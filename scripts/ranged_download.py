"""Parallel HTTPS ranges with resume and whole-file SHA-256 verification."""
import concurrent.futures,hashlib,os,threading,time
from pathlib import Path
import requests
_local=threading.local()
def session():
    if not hasattr(_local,'session'):_local.session=requests.Session()
    return _local.session
def download_files(files,workers=24,block=32*1024*1024):
    pending=[];jobs=[]
    for entry in files:
        p=Path(entry['path']);p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.stat().st_size==entry['size']:
            if not entry.get('sha256') or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==entry['sha256']:
                continue
        temp=p.with_name(p.name+'.partial')
        if p.exists() and not temp.exists():p.rename(temp)
        if not temp.exists():temp.touch()
        with temp.open('r+b') as f:f.truncate(entry['size'])
        # Resolve redirects once; share the signed CDN URL between ranges.
        with session().get(entry['url'],headers={'Range':'bytes=0-0'},stream=True,timeout=(20,60)) as r:
            r.raise_for_status();entry['download_url']=r.url
        pending.append((entry,p,temp))
        with temp.open('rb') as f:
            for start in range(0,entry['size'],block):
                end=min(start+block,entry['size'])-1
                if os.lseek(f.fileno(),start,os.SEEK_HOLE)>=end+1:continue
                jobs.append((entry,temp,start,end))
    def fetch(job):
        entry,temp,start,end=job
        for attempt in range(8):
            try:
                with session().get(entry['download_url'],headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,45)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{entry["size"]}':
                        raise ValueError('Incorrect byte-range response')
                    count=0
                    with temp.open('r+b') as f:
                        f.seek(start)
                        for chunk in r.iter_content(1024*1024):
                            f.write(chunk);count+=len(chunk)
                    if count!=end-start+1:raise ValueError('Incomplete range')
                return count
            except Exception:
                if attempt==7:raise
                time.sleep(min(attempt+1,5))
    started=time.monotonic();done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch,j) for j in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            done+=f.result()
            if i%16==0 or i==len(jobs):print(f'RANGES {i}/{len(jobs)} {done/1e9:.2f}GB {done/1e6/max(time.monotonic()-started,1):.1f}MB/s',flush=True)
    for entry,p,temp in pending:
        if entry.get('sha256') and hashlib.file_digest(temp.open('rb'),'sha256').hexdigest()!=entry['sha256']:
            temp.unlink();raise ValueError(f'Checkpoint SHA-256 mismatch: {p.name}')
        temp.rename(p)
        p.with_name(p.name+'.aria2').unlink(missing_ok=True)
        print('VERIFIED',p.name,flush=True)
