from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from arena import evaluate
root=Path(__file__).parents[1]; out=root/'results'; out.mkdir(exist_ok=True); result=evaluate(5)
(out/'metrics.json').write_text(json.dumps({'seed_count':result['seed_count'],'dataset':result['dataset'],'summary':result['summary']},indent=2),encoding='utf-8')
with (out/'raw.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=result['raw'][0].keys()); writer.writeheader(); writer.writerows(result['raw'])
print(json.dumps({'seed_count':result['seed_count'],'dataset':result['dataset'],'summary':result['summary']},indent=2))
