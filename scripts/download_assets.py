"""Download official kitchen assets independently of simulator imports."""
import concurrent.futures,json,os,subprocess,zipfile
from pathlib import Path
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
    subprocess.run(['curl','--fail','--location','--retry','8','--retry-delay','3',
                    '--continue-at','-','--output',str(archive),url],check=True)
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
