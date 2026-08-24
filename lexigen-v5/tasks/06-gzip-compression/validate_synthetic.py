from __future__ import annotations
import argparse, gzip, json, math, os, random
from pathlib import Path
from candidates import CANDIDATES_BY_ARM, CANDIDATE_META, reference_exact


def cases():
    rng = random.Random(20260824)
    random_bytes = bytes(rng.randrange(256) for _ in range(32768))
    text = (b"alpha beta gamma delta epsilon " * 6000)[:131072]
    repeated = (b"A" * 65536) + (b"BCDE" * 16384)
    structured = bytes((i // 32) % 256 for i in range(131072))
    mixed = (bytes(range(256)) * 128) + (b"compression compression gzip " * 3000)
    return [
        {"name":"empty","plaintext":b""},
        {"name":"tiny","plaintext":b"hello gzip"},
        {"name":"repeated","plaintext":repeated},
        {"name":"text_zipf_proxy","plaintext":text},
        {"name":"structured_image_proxy","plaintext":structured},
        {"name":"mixed_random","plaintext":mixed + random_bytes},
    ]


def verify(problem, solution, reference):
    if not isinstance(solution, dict) or not isinstance(solution.get("compressed_data"), bytes):
        return False, "format"
    data = bytes(solution["compressed_data"])
    try:
        plain = gzip.decompress(data)
    except Exception as e:
        return False, f"decompress:{type(e).__name__}"
    if plain != problem["plaintext"]:
        return False, "plaintext_mismatch"
    ref_len = len(reference["compressed_data"])
    if len(data) > math.ceil(ref_len * 1.001):
        return False, f"size:{len(data)}>{math.ceil(ref_len*1.001)}"
    return True, None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    rows=[]
    for ci, problem in enumerate(cases(),1):
        ref=reference_exact(problem)
        for arm, candidates in CANDIDATES_BY_ARM.items():
            for name, fn in candidates:
                try:
                    got=fn(problem); ok, reason=verify(problem,got,ref); err=None
                except Exception as e:
                    got=None; ok=False; reason="exception"; err=f"{type(e).__name__}: {e}"
                rows.append({"case_index":ci,"case":problem["name"],"plaintext_bytes":len(problem["plaintext"]),"arm":arm,"candidate":name,"implementation_class":CANDIDATE_META[name]["implementation_class"],"valid":bool(ok),"failure_reason":err or reason,"compressed_bytes":len(got["compressed_data"]) if isinstance(got,dict) and isinstance(got.get("compressed_data"),bytes) else None,"reference_bytes":len(ref["compressed_data"])})
    if len(rows)!=180: raise RuntimeError(f"expected 180 checks got {len(rows)}")
    eligible=[]
    for arm,candidates in CANDIDATES_BY_ARM.items():
        for name,_ in candidates:
            rs=[r for r in rows if r["candidate"]==name]
            if all(r["valid"] for r in rs): eligible.append(name)
    report={"campaign":"LEXIGEN v5 Causal Transfer Generalization Experiment","task_index":6,"task":"gzip_compression","stage":"synthetic_r1","checks":len(rows),"valid_checks":sum(r["valid"] for r in rows),"candidate_count":30,"eligible_candidate_count":len(eligible),"eligible_candidates":eligible,"official_training_manifest_opened":False,"official_training_payloads_opened":0,"official_test_manifest_opened":False,"official_test_payloads_opened":0,"threshold_changes":False}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/"synthetic-summary.json").write_text(json.dumps(report,indent=2)+"\n");(a.output/"synthetic-results.jsonl").write_text("\n".join(json.dumps(r,separators=(",",":")) for r in rows)+"\n");print(json.dumps(report,indent=2))

if __name__=="__main__": main()
