"""Download official kitchen assets independently of simulator imports."""
import concurrent.futures,json,os,subprocess,zipfile,shutil
from pathlib import Path
import requests
from ranged_download import download_files
root=Path(os.environ.get('ROBOCASA_ROOT','/root/autodl-tmp/robocasa-repro'))
assets=root/'repos/robocasa/robocasa/models/assets'
links=json.loads((assets/'box_links/box_links_assets.json').read_text())
mapping={'textures':'textures','generative_textures':'generative_textures',
         'fixtures_lightwheel':'fixtures','objaverse':'objects/objaverse',
         'aigen_objs':'objects/aigen_objs','objects_lightwheel':'objects/lightwheel'}
def get(item):
    key,target=item
    marker=root/'data/asset_status'/f'{key}.json'
    if marker.exists():return
    url=links[key].replace('/s/','/shared/static/')+'.zip'
    archive=root/'data/downloads'/f'{key}.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if os.environ.get('REPRO_ASSET_RANGES','1')=='1':
        with requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=(20,60)) as r:
            r.raise_for_status()
            size=int(r.headers['Content-Range'].rsplit('/',1)[1])
        download_files([{'path':str(archive),'url':url,'size':size}],workers=16)
        command=None
    elif shutil.which('aria2c'):
        command=['aria2c','--continue=true','--max-connection-per-server=8','--split=8',
                 '--min-split-size=16M','--file-allocation=none','--max-tries=8',
                 '--retry-wait=3','--summary-interval=30','--console-log-level=warn',
                 '--dir',str(archive.parent),'--out',archive.name,url]
    else:
        command=['curl','--fail','--location','--retry','8','--retry-delay','3',
                 '--continue-at','-','--output',str(archive),url]
    if command:subprocess.run(command,check=True)
    destination=(assets/target).parent
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (destination/name).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Archive path leaves asset directory')
        z.extractall(destination)
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps({'source':url,'archive':str(archive),'destination':str(destination)}))
    print('ASSET_READY',key,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(get,mapping.items()))
