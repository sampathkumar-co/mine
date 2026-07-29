from __future__ import annotations
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def load(name): return json.loads((HERE/name).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def check(v,m):
    if not v: raise AssertionError(m)
def test_registry_and_boundary():
    pre=load('V19R5_PRECOMMIT.json'); reg=load('V19R5_REGISTRY.json'); report=load('V19R5_DISCOVERY_REPORT.json')
    check(sha(HERE/'V19R5_REGISTRY.json')==pre['registry_sha256'],'registry hash')
    check(reg['discovery_task_count']==631 and reg['validation_task_count']==244,'split counts')
    check(report['discovery_tasks']==631 and report['discovery_generators_imported']==631,'discovery denominator')
    check(report['validation_generators_imported']==0 and not report['validation_outputs_opened'],'validation boundary')
def test_discovery_counts():
    report=load('V19R5_DISCOVERY_REPORT.json'); check(report['generator_invalid_tasks']==10,'invalid tasks'); check(report['total_exact_complete_programs']==1760,'program count'); check(report['distinct_structures_seen']==8,'structure count'); check(report['qualifying_productions']==5,'library count')
def test_library_integrity():
    report=load('V19R5_DISCOVERY_REPORT.json'); library=load('V19R5_LIBRARY.json'); check(sha(HERE/'V19R5_LIBRARY.json')==report['library_sha256'],'library hash'); check(library['qualifying_production_count']==5,'qualifying count'); check(not library['validation_outputs_opened'],'validation flag')
    seen=set()
    all_task_ids=set(load('V19R5_REGISTRY.json')['discovery_task_ids']+load('V19R5_REGISTRY.json')['validation_task_ids'])
    for item in library['productions']:
        ph=item['production_sha256']; check(ph not in seen,'duplicate production'); seen.add(ph); check(item['discovery_task_count']>=2,'weak qualification'); check(len(item['discovery_task_ids'])==item['discovery_task_count'],'task count mismatch')
        path=HERE/'library'/f'production-{ph}.json'; check(path.exists(),'missing production'); check(sha(path)==item['production_file_sha256'],'production file hash'); text=path.read_text(encoding='utf-8'); check(not any(task in text for task in all_task_ids),'task id leaked')
def test_structure_uniqueness():
    library=load('V19R5_LIBRARY.json'); structures=[json.dumps(item['structure'],sort_keys=True,separators=(',',':')) for item in library['productions']]; check(len(structures)==len(set(structures))==5,'structure uniqueness')
def main():
    tests=[v for n,v in sorted(globals().items()) if n.startswith('test_') and callable(v)]
    for test in tests: test(); print('PASS',test.__name__)
    print(f'SUMMARY {len(tests)}/{len(tests)} tests passed')
if __name__=='__main__': main()
