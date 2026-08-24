from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from parallel_candidates import ARM_ORDER, REFERENCE_SOLVERS, edge_direct, flat_candidates

REVISION="bb02811fa47ca1c833baaa344949bcd8fb307ac8"
SOURCE_COMMIT="dff9914c10800c7a031c9e8c3d4d1c8cd1b38906"
SHARDS=5
EXPECTED_RECORDS=100

TASKS={
 "odr":{"index":2,"family":"miscellaneous","source_path":"AlgoTuneTasks/odr/odr.py","source_sha256":"076efd6697175397912d5d8e3bc1b121ba7461db3fdbf04263fa6d57f81eb68c","applicable":["TM-BFR-01","TM-CAC-01","TM-PBEB-01","TM-RRR-01"]},
 "dct_type_I_scipy_fftpack":{"index":3,"family":"signal_processing","source_path":"AlgoTuneTasks/dct_type_I_scipy_fftpack/dct_type_I_scipy_fftpack.py","source_sha256":"d9667553f833e9966df0d6fde154c473f7b33285fd67a28f216f9c3df25d4e11","applicable":["TM-BFR-01","TM-CAC-01","TM-RRR-01"]},
 "ode_lorenz96_nonchaotic":{"index":4,"family":"scientific_computing","source_path":"AlgoTuneTasks/ode_lorenz96_nonchaotic/ode_lorenz96_nonchaotic.py","source_sha256":"629456c15f57d932ded725c14d468bee2a513b97d124ad95d38401841bf68621","applicable":["TM-BFR-01","TM-CAC-01"]},
 "robust_kalman_filter":{"index":5,"family":"signal_processing","source_path":"AlgoTuneTasks/robust_kalman_filter/robust_kalman_filter.py","source_sha256":"3c589ba1f0d988f1d89db7a21d2f28a6d588334f881a407fbedc1a2c15a5bec2","applicable":["TM-BFR-01","TM-CAC-01","TM-RRR-01"]},
 "edge_expansion":{"index":6,"family":"miscellaneous","source_path":"AlgoTuneTasks/edge_expansion/edge_expansion.py","source_sha256":"24d194fbf8f604d318b9f330e61ad084ff4ea498de2c0a299835ad7ecce55d9a","applicable":["TM-BFR-01","TM-CAC-01","TM-RRR-01"]},
 "svd":{"index":7,"family":"linear_algebra","source_path":"AlgoTuneTasks/svd/svd.py","source_sha256":"8f7771e5618509c0b8af73390440dd7258a253983cdf8cc03b0208fa4218b018","applicable":["TM-BFR-01","TM-CAC-01","TM-PBEB-01"]},
 "max_independent_set_cpsat":{"index":8,"family":"combinatorial","source_path":"AlgoTuneTasks/max_independent_set_cpsat/max_independent_set_cpsat.py","source_sha256":"c81c015833ed76feb13b0b9e630b6fb5b44b83de64fc9e32facffd45c5fb86fb","applicable":["TM-BFR-01","TM-CAC-01","TM-RRR-01"]},
}


def fetch(url: str, ua="LEXIGEN-v6-parallel7") -> bytes:
    last=None
    for attempt in range(8):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":ua})
            with urllib.request.urlopen(req,timeout=240) as r: return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as exc:
            last=exc; time.sleep(min(45,2**attempt))
    raise RuntimeError(f"fetch exhausted {url}") from last


def decode_value(value, base: str):
    if isinstance(value,list): return [decode_value(x,base) for x in value]
    if not isinstance(value,dict): return value
    kind=value.get("__type__")
    if kind is None: return {k:decode_value(v,base) for k,v in value.items()}
    if kind=="ndarray_ref":
        rel=str(value.get("npy_path",""))
        if not rel or rel.startswith("/") or ".." in Path(rel).parts: raise RuntimeError(f"unsafe ndarray_ref {rel}")
        return np.load(io.BytesIO(fetch(f"{base}/{rel}?download=true")),allow_pickle=False)
    if kind=="ndarray_b64":
        raw=base64.b64decode(str(value.get("data_b64","")).encode("ascii")); arr=np.frombuffer(raw,dtype=np.dtype(value["dtype"])); shape=tuple(value.get("shape",[])); return arr.reshape(shape) if shape else arr
    if kind=="ndarray": return np.array(value["data"],dtype=np.dtype(value.get("dtype")))
    if kind=="tuple": return tuple(decode_value(x,base) for x in value.get("data",[]))
    return {k:decode_value(v,base) for k,v in value.items() if k!="__type__"}


def timed(fn,problem):
    try:
        t=time.perf_counter_ns(); out=fn(problem); return out,time.perf_counter_ns()-t,None
    except Exception as exc: return None,None,f"{type(exc).__name__}: {exc}"


def _kalman_obj(problem,sol):
    w=np.asarray(sol["w_hat"],dtype=float); v=np.asarray(sol["v_hat"],dtype=float); tau=float(problem["tau"]); M=float(problem["M"])
    J=float(np.sum(w*w))
    for row in v:
        val=float(np.linalg.norm(row)); J += tau*(val*val if val<=M else 2*M*val-M*M)
    return J


def verify(task,problem,got,ref):
    try:
        if task=="odr":
            beta=np.asarray(got["beta"],dtype=float); expected=np.asarray(ref["beta"],dtype=float)
            rtol=2*np.finfo(float).eps**(2/3); atol=np.finfo(float).smallest_normal
            ok=beta.shape==(2,) and np.all(np.isfinite(beta)) and np.allclose(beta,expected,rtol=rtol,atol=atol)
            return ok,None if ok else "odr_beta_mismatch",{}
        if task=="dct_type_I_scipy_fftpack":
            a=np.asarray(got,dtype=float); b=np.asarray(ref,dtype=float)
            if a.shape!=b.shape or not np.all(np.isfinite(a)): return False,"dct_shape_or_nonfinite",{}
            err=float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12)); return err<=1e-6,None if err<=1e-6 else "dct_relative_error",{"relative_error":err}
        if task=="ode_lorenz96_nonchaotic":
            a=np.asarray(got,dtype=float); b=np.asarray(ref,dtype=float); y0=np.asarray(problem["y0"],dtype=float)
            ok=a.shape==y0.shape and np.all(np.isfinite(a)) and np.allclose(a,b,rtol=1e-5,atol=1e-8)
            return ok,None if ok else "ode_final_state_mismatch",{"max_abs_error":float(np.max(np.abs(a-b))) if a.shape==b.shape else math.inf}
        if task=="robust_kalman_filter":
            req={"x_hat","w_hat","v_hat"}
            if not isinstance(got,dict) or not req.issubset(got): return False,"kalman_format",{}
            A=np.asarray(problem["A"],dtype=float); B=np.asarray(problem["B"],dtype=float); C=np.asarray(problem["C"],dtype=float); y=np.asarray(problem["y"],dtype=float); x0=np.asarray(problem["x_initial"],dtype=float)
            x=np.asarray(got["x_hat"],dtype=float); w=np.asarray(got["w_hat"],dtype=float); v=np.asarray(got["v_hat"],dtype=float); N,m=y.shape; n=A.shape[1]; p=B.shape[1]
            if x.shape!=(N+1,n) or w.shape!=(N,p) or v.shape!=(N,m) or not(np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(v).all()): return False,"kalman_shape_nonfinite",{}
            if np.linalg.norm(x[0]-x0)>1e-5: return False,"kalman_initial",{}
            for t in range(N):
                if np.linalg.norm(x[t+1]-(A@x[t]+B@w[t]))>1e-5: return False,"kalman_dynamics",{}
                if np.linalg.norm(y[t]-(C@x[t]+v[t]))>1e-5: return False,"kalman_measurement",{}
            J=_kalman_obj(problem,got); Jref=_kalman_obj(problem,ref); ok=J<=Jref*1.01+1e-8
            return ok,None if ok else "kalman_suboptimal",{"candidate_objective":J,"reference_objective":Jref}
        if task=="edge_expansion":
            val=float(got["edge_expansion"]); expected=float(edge_direct(problem)["edge_expansion"]); ok=math.isfinite(val) and val>=0 and math.isclose(val,expected,rel_tol=1e-5,abs_tol=1e-8)
            return ok,None if ok else "edge_expansion_mismatch",{"candidate":val,"expected":expected}
        if task=="svd":
            if not isinstance(got,dict) or not {"U","S","V"}.issubset(got): return False,"svd_format",{}
            A=np.asarray(problem["matrix"] if isinstance(problem,dict) else problem,dtype=float); U=np.asarray(got["U"],dtype=float); s=np.asarray(got["S"],dtype=float); V=np.asarray(got["V"],dtype=float); n,m=A.shape; k=min(n,m)
            if U.shape!=(n,k) or s.shape!=(k,) or V.shape!=(m,k) or not(np.isfinite(U).all() and np.isfinite(s).all() and np.isfinite(V).all()): return False,"svd_shape_nonfinite",{}
            if np.any(s<0) or not np.allclose(U.T@U,np.eye(k),atol=1e-6) or not np.allclose(V.T@V,np.eye(k),atol=1e-6): return False,"svd_orthogonality",{}
            err=float(np.max(np.abs(A-U@np.diag(s)@V.T))); ok=np.allclose(A,U@np.diag(s)@V.T,atol=1e-6)
            return bool(ok),None if ok else "svd_reconstruction",{"max_abs_error":err}
        if task=="max_independent_set_cpsat":
            n=len(problem); sol=[int(x) for x in got]
            if len(sol)!=len(set(sol)) or any(x<0 or x>=n for x in sol): return False,"mis_duplicate_or_range",{}
            for i,u in enumerate(sol):
                for v in sol[i+1:]:
                    if int(problem[u][v])==1: return False,"mis_not_independent",{}
            ok=len(sol)==len(ref); return ok,None if ok else "mis_not_optimal",{"candidate_size":len(sol),"optimal_size":len(ref)}
        raise KeyError(task)
    except Exception as exc: return False,f"verify_exception:{type(exc).__name__}:{exc}",{}


def synthetic_problems(task):
    rng=np.random.default_rng(20260824)
    if task=="odr":
        return [
            {"x":[1.,2.,4.,8.,16.,32.],"y":[2.1,3.0,5.2,9.0,17.1,33.2],"sx":[.1,.12,.15,.2,.3,.5],"sy":[.15,.13,.2,.25,.35,.6]},
            {"x":[.5,1.5,3.,6.,12.,24.,48.],"y":[1.0,2.0,3.7,7.2,14.1,28.0,56.2],"sx":[.05,.08,.12,.2,.4,.7,1.],"sy":[.07,.1,.15,.3,.5,.8,1.2]},
        ]
    if task=="dct_type_I_scipy_fftpack": return [rng.normal(size=(5,5)),rng.normal(size=(9,7))]
    if task=="ode_lorenz96_nonchaotic": return [{"F":2.0,"t0":0.0,"t1":10.0,"y0":(2+rng.random(8)*.01).tolist()},{"F":2.0,"t0":0.0,"t1":6.0,"y0":(2+rng.random(12)*.01).tolist()}]
    if task=="robust_kalman_filter":
        out=[]
        for N in (4,5):
            A=np.array([[.85,.05],[0.,.8]]); B=np.array([[.2],[.1]]); C=np.eye(2); x0=np.array([.2,-.1]); y=np.vstack([x0+np.array([.03*i,-.02*i]) for i in range(N)])
            out.append({"A":A.tolist(),"B":B.tolist(),"C":C.tolist(),"y":y.tolist(),"x_initial":x0.tolist(),"tau":1.0,"M":3.0})
        return out
    if task=="edge_expansion": return [{"adjacency_list":[[1,2],[2],[0,3],[1]],"nodes_S":[0,1]},{"adjacency_list":[[1],[2],[3],[0]],"nodes_S":[1]}]
    if task=="svd": return [{"matrix":rng.normal(size=(6,4))},{"matrix":rng.normal(size=(5,8))}]
    if task=="max_independent_set_cpsat": return [
        [[0,1,0,1,0],[1,0,1,0,0],[0,1,0,1,1],[1,0,1,0,0],[0,0,1,0,0]],
        [[0,1,1,0,0,0],[1,0,0,1,0,0],[1,0,0,0,1,0],[0,1,0,0,0,1],[0,0,1,0,0,1],[0,0,0,1,1,0]],
    ]
    raise KeyError(task)


def pretrain_certificate(task,source_text):
    candidates=flat_candidates(task,source_text); by_class={}
    for c in candidates: by_class.setdefault(c.implementation_class,c)
    rows=[]
    for case_idx,problem in enumerate(synthetic_problems(task),1):
        ref=REFERENCE_SOLVERS[task](problem)
        for impl,c in by_class.items():
            got=c.solve(problem); ok,reason,metrics=verify(task,problem,got,ref)
            rows.append({"case":case_idx,"implementation_class":impl,"representative":c.name,"valid":bool(ok),"reason":reason,**metrics})
            if not ok: raise RuntimeError(f"{task} synthetic failed class={impl} case={case_idx}: {reason}")
    return candidates,rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task",choices=sorted(TASKS),required=True); ap.add_argument("--shard",type=int,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    if not 0<=args.shard<SHARDS: raise ValueError("invalid shard")
    task=args.task; cfg=TASKS[task]
    lock=json.loads(Path("lexigen-v6/parallel7/PARALLEL7_LOCK.json").read_text())
    if lock["official_training_manifests_opened_before_lock"] or lock["official_test_manifests_opened_before_lock"] or lock["official_payloads_opened_before_lock"]: raise RuntimeError("parallel7 data boundary crossed")
    if not lock["execution_deduplication_by_frozen_implementation_class"]: raise RuntimeError("dedup policy mismatch")
    src_url=f"https://raw.githubusercontent.com/oripress/AlgoTune/{SOURCE_COMMIT}/{cfg['source_path']}"; src_raw=fetch(src_url); src_sha=hashlib.sha256(src_raw).hexdigest()
    if src_sha!=cfg["source_sha256"]: raise RuntimeError(f"source sha mismatch {task} {src_sha}")
    source_text=src_raw.decode("utf-8")
    candidates,synthetic_rows=pretrain_certificate(task,source_text)
    class_members=defaultdict(list)
    for c in candidates: class_members[c.implementation_class].append(c)
    reps={impl:members[0] for impl,members in class_members.items()}

    base=f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{task}"
    train_name=f"{task}_T100ms_n64_size100_train.jsonl"; test_name=f"{task}_T100ms_n64_size100_test.jsonl"
    manifest=fetch(f"{base}/{train_name}?download=true"); manifest_sha=hashlib.sha256(manifest).hexdigest(); records=[json.loads(line) for line in manifest.decode("utf-8").splitlines() if line.strip()]
    if len(records)!=EXPECTED_RECORDS: raise RuntimeError(f"{task}: expected 100 train records got {len(records)}")
    evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(records) if i%SHARDS==args.shard):
        problem=decode_value(row["problem"],base)
        classes=list(reps); shift=idx%len(classes); classes=classes[shift:]+classes[:shift]
        if idx%2==0:
            ref,ref_ns,ref_err=timed(REFERENCE_SOLVERS[task],problem); class_runs={impl:timed(reps[impl].solve,problem) for impl in classes}; execution_order="reference_first"
        else:
            class_runs={impl:timed(reps[impl].solve,problem) for impl in classes}; ref,ref_ns,ref_err=timed(REFERENCE_SOLVERS[task],problem); execution_order="classes_first"
        if ref_err or ref is None or ref_ns is None: raise RuntimeError(f"{task} reference failed record={idx+1}: {ref_err}")
        for impl,members in class_members.items():
            got,c_ns,c_err=class_runs[impl]
            if c_err is None: valid,reason,metrics=verify(task,problem,got,ref)
            else: valid,reason,metrics=False,"exception",{}
            for c in members:
                evidence.append({
                    "task":task,"task_index":cfg["index"],"family":cfg["family"],"index":idx+1,"seed":int(row.get("seed",idx+1)),
                    "arm":c.arm,"candidate":c.name,"implementation_class":c.implementation_class,"operators":list(c.operators),"transfer_ids":list(c.transfer_ids),"learned_template":c.learned_template,"baseline_id":c.baseline_id,
                    "valid":bool(valid and c_err is None),"failure_reason":c_err or reason,"candidate_ns":c_ns,"reference_ns":ref_ns,"speedup":(ref_ns/c_ns) if c_ns and c_ns>0 else 0.0,
                    "shared_execution_class":True,"class_candidate_count":len(members),"execution_order":execution_order,"shard":args.shard,"invalid_output_retries":0,
                    "train_manifest_name":train_name,"train_manifest_sha256":manifest_sha,"expected_test_manifest_name":test_name,"source_sha256":src_sha,"test_manifest_contents_opened":False,"test_payloads_opened":0,
                    **metrics,
                })
        del problem,ref,class_runs; gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text("\n".join(json.dumps(x,separators=(",",":")) for x in evidence)+"\n")
    cert_sha=hashlib.sha256(json.dumps(synthetic_rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    print(json.dumps({"task":task,"shard":args.shard,"candidate_rows":len(evidence),"implementation_classes":sorted(reps),"synthetic_certificate_sha256":cert_sha,"train_manifest_sha256":manifest_sha,"test_opened":False},indent=2))

if __name__=="__main__": main()
