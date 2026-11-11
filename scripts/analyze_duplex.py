"""Summarize measured concurrency, cancellation and recording from event logs."""
import argparse,json,statistics
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
events=[json.loads(line) for line in (args.run/'duplex_events.jsonl').read_text().splitlines()]
controls=[e for e in events if e['event']=='control_step']
vlm=[];vla=[];tools=[]
for e in events:
    if e['event'] in {'vlm_end','vla_end','tool_prepare_end'}:
        duration=e['duration_s'];start=e['monotonic']-duration;end=e['monotonic']
        steps=[c for c in controls if start<c['monotonic']<=end]
        row={'duration_s':duration,'control_steps_during_inference':len(steps),
             'moving_steps_during_inference':sum(c['executor']!='hold' for c in steps)}
        if e['event']=='vlm_end': row['decision']=e['decision'];vlm.append(row)
        elif e['event']=='vla_end':vla.append(row)
        else:row['tool']=e['tool'];tools.append(row)
latencies=[e['delivery_latency_s'] for e in events if e['event']=='command_applied']
summary={'vlm_calls':vlm,'vla_calls':len(vla),'tool_preparations':tools,
    'vlm_mean_s':statistics.mean(r['duration_s'] for r in vlm) if vlm else None,
    'vla_mean_s':statistics.mean(r['duration_s'] for r in vla) if vla else None,
    'vla_motion_overlap_steps':sum(r['moving_steps_during_inference'] for r in vla),
    'max_command_delivery_s':max(latencies,default=None),
    'obsolete_results_discarded':sum(e['event']=='result_discarded' and e.get('reason')=='obsolete_revision' for e in events),
    'sam3_calls':sum(e['event']=='sam3_result' for e in events),
    'graspgen_calls':sum(e['event']=='graspgen_result' for e in events),
    'result':json.loads((args.run/'result.json').read_text())}
(args.run/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps(summary,indent=2,ensure_ascii=False))
