from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from . import conditioned_cell_frontier_v60 as conditioned
from . import external_response_cost_v58 as external
from . import response_cost_export_v57 as export_v57
from . import response_cost_pareto_v56 as response


MANIFEST = Path(__file__).resolve().parents[2] / "external-data" / "uci-v61" / "manifest.json"
PROFILE_SEEDS = conditioned.PROFILE_SEEDS
BUDGET_LADDER = conditioned.BUDGET_LADDER
MAX_RECORDS = 384


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent":"Mini-ORIGIN-v0.61-evaluation/1"})
    with urlopen(request, timeout=240) as handle:
        return handle.read()


def member(archive: zipfile.ZipFile, basename: str) -> str:
    matches=[name for name in archive.namelist() if name.rsplit("/",1)[-1].lower()==basename.lower()]
    if len(matches)!=1:
        raise RuntimeError(f"expected one {basename!r}, found {matches!r}; archive={archive.namelist()!r}")
    return matches[0]


def nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def comma_rows(text: str) -> list[list[str]]:
    return [[value.strip() for value in line.split(",")] for line in nonempty_lines(text)]


def whitespace_rows(text: str) -> list[list[str]]:
    return [line.split() for line in nonempty_lines(text)]


def parse_label_first(rows: list[list[str]]) -> list[tuple[tuple[str,...],str]]:
    return [(tuple(row[1:]),row[0]) for row in rows if len(row)>=2]


def parse_label_last(rows: list[list[str]]) -> list[tuple[tuple[str,...],str]]:
    return [(tuple(row[:-1]),row[-1]) for row in rows if len(row)>=2]


def parse_records(name: str, payload: bytes) -> list[tuple[tuple[str,...],str]]:
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        if name=="Soybean (Large)":
            rows=[]
            for basename in ("soybean-large.data","soybean-large.test"):
                text=archive.read(member(archive,basename)).decode("utf-8",errors="replace")
                rows.extend(comma_rows(text))
            records=parse_label_first(rows)
        elif name=="Annealing":
            rows=[]
            for basename in ("anneal.data","anneal.test"):
                text=archive.read(member(archive,basename)).decode("utf-8",errors="replace")
                rows.extend(comma_rows(text))
            records=parse_label_last(rows)
        elif name=="Primary Tumor":
            text=archive.read(member(archive,"primary-tumor.data")).decode("utf-8",errors="replace")
            records=parse_label_first(comma_rows(text))
        elif name=="Optical Recognition of Handwritten Digits":
            rows=[]
            for basename in ("optdigits.tra","optdigits.tes"):
                text=archive.read(member(archive,basename)).decode("utf-8",errors="replace")
                rows.extend(comma_rows(text))
            records=parse_label_last(rows)
        elif name=="Statlog (Landsat Satellite)":
            rows=[]
            for basename in ("sat.trn","sat.tst"):
                text=archive.read(member(archive,basename)).decode("utf-8",errors="replace")
                rows.extend(whitespace_rows(text))
            records=parse_label_last(rows)
        elif name=="Madelon":
            records=[]
            for split in ("train","valid"):
                data_text=archive.read(member(archive,f"madelon_{split}.data")).decode("utf-8",errors="replace")
                label_text=archive.read(member(archive,f"madelon_{split}.labels")).decode("utf-8",errors="replace")
                data_rows=whitespace_rows(data_text)
                labels=nonempty_lines(label_text)
                if len(data_rows)!=len(labels):
                    raise RuntimeError(f"Madelon {split} row/label mismatch")
                records.extend((tuple(row),label) for row,label in zip(data_rows,labels))
        elif name=="Glioma Grading Clinical and Mutation Features":
            text=archive.read(member(archive,"TCGA_InfoWithGrade.csv")).decode("utf-8-sig",errors="replace")
            reader=csv.DictReader(StringIO(text))
            if not reader.fieldnames or "Grade" not in reader.fieldnames:
                raise RuntimeError(f"Glioma fields missing Grade: {reader.fieldnames!r}")
            excluded={"Grade","Project","Case_ID","Primary_Diagnosis","","Unnamed: 0"}
            feature_names=[field for field in reader.fieldnames if field not in excluded and not field.startswith("Unnamed")]
            records=[(tuple((row.get(field) or "?").strip() for field in feature_names),(row.get("Grade") or "?").strip()) for row in reader]
        else:
            raise KeyError(name)
    widths={len(features) for features,_ in records}
    if not records or len(widths)!=1:
        raise RuntimeError(f"bad records for {name}: count={len(records)} widths={widths}")
    return records


def compact_state(task: object, allowed: int, remaining: int, seed: int) -> dict[str,object]:
    row=export_v57.compact_state(task,allowed,remaining,seed)
    base_digest=hashlib.sha256(f"v61:{task.name}:{allowed}:{remaining}".encode("utf-8")).hexdigest()
    row["base_digest"]=base_digest
    row["digest"]=hashlib.sha256(f"{base_digest}:{seed}:external-conditioned-v61".encode("utf-8")).hexdigest()
    return row


def run(states_path: Path, reference_path: Path) -> dict[str,object]:
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks=[]
    summaries=[]
    verification=[]
    for dataset in manifest["datasets"]:
        payload=download(str(dataset["url"]))
        actual_hash=hashlib.sha256(payload).hexdigest()
        matched=actual_hash==dataset["sha256"] and len(payload)==dataset["bytes"]
        verification.append({"name":dataset["name"],"expected_sha256":dataset["sha256"],"actual_sha256":actual_hash,"expected_bytes":dataset["bytes"],"actual_bytes":len(payload),"matched":matched})
        if not matched:
            raise RuntimeError(f"archive mismatch for {dataset['name']}")
        records=parse_records(str(dataset["name"]),payload)
        task,summary=external.task_from_records(str(dataset["name"]),records)
        selected,selection=conditioned.select_states(task)
        summary.update(selection)
        summary["task"]=task.name
        summaries.append(summary)
        tasks.append((task,selected))

    state_rows=[]
    reference_rows=[]
    base_states=set()
    for task,selected in tasks:
        for allowed,remaining,representatives in selected:
            base_digest=hashlib.sha256(f"v61:{task.name}:{allowed}:{remaining}".encode("utf-8")).hexdigest()
            base_states.add(base_digest)
            for seed in PROFILE_SEEDS:
                profile=response.profile_for_task(task,seed)
                result=response.evaluate_state(task,profile,allowed,remaining)
                compact=compact_state(task,allowed,remaining,seed)
                state_rows.append(compact)
                result.update({"task":task.name,"base_state_digest":base_digest,"state_digest":compact["digest"],"structural_partition_representatives":representatives})
                reference_rows.append(result)
    state_rows.sort(key=lambda row:str(row["digest"]))
    reference_rows.sort(key=lambda row:str(row["state_digest"]))
    export_v57.write_text(state_rows,states_path)

    solved=[row for row in reference_rows if row["pareto_solved"]]
    both=[row for row in solved if row["plain_solved"]]
    pareto_only=[row for row in solved if not row["plain_solved"]]
    ratios=[float(row["expansion_ratio_lower_bound"]) for row in solved]
    ladder={str(budget):{key:sum(int(row["budget_ladder"][str(budget)][key]) for row in reference_rows) for key in ("pareto_solved","plain_solved")} for budget in BUDGET_LADDER}
    payload={
        "status":"external_conditioned_python_reference_v61",
        "parent_v60_digest":"fe0c56d83095bfc6607f9bfffb480ca829f369d6bd9db5ff47860184f7e246fb",
        "archive_lock_digest":manifest["lock_digest"],
        "archive_verification":{"all_hashes_match":all(row["matched"] for row in verification),"rows":verification},
        "protocol":json.loads(json.dumps({"profile_seeds":list(conditioned.PROFILE_SEEDS),"path_seeds":list(conditioned.PATH_SEEDS),"maximum_depth":conditioned.MAX_DEPTH,"maximum_query_choices":conditioned.MAX_QUERY_CHOICES,"maximum_cells_per_depth":conditioned.MAX_CELLS_PER_DEPTH,"sample_sizes":list(conditioned.SAMPLE_SIZES),"max_states_per_task":conditioned.MAX_STATES_PER_TASK,"partition_class_range":[conditioned.MIN_PARTITION_CLASSES,conditioned.MAX_PARTITION_CLASSES],"raw_query_range":[conditioned.MIN_RAW_QUERIES,conditioned.MAX_RAW_QUERIES],"minimum_redundancy":conditioned.MIN_REDUNDANCY,"budget":response.BUDGET,"budget_ladder":list(BUDGET_LADDER)})),
        "dataset_summaries":summaries,
        "contributing_dataset_count":sum(int(row["selected_states"]>0) for row in summaries),
        "base_state_count":len(base_states),
        "profiled_state_count":len(reference_rows),
        "pareto_solved_count":len(solved),
        "both_solved_count":len(both),
        "pareto_only_count":len(pareto_only),
        "plan_mismatch_count":sum(int(not row["matched_if_both"]) for row in both),
        "dominated_queries_removed":sum(int(row["pareto_stats"]["dominated_queries_removed"]) for row in solved),
        "root_incomparable_classes":sum(int(row["root_pareto_certificate"]["incomparable_pareto_classes"]) for row in reference_rows),
        "expansion_ratio_median":float(np.median(ratios)) if ratios else None,
        "expansion_ratio_p90":float(np.quantile(ratios,0.9)) if ratios else None,
        "budget_ladder_summary":ladder,
        "state_input_sha256":hashlib.sha256(states_path.read_bytes()).hexdigest(),
        "rows":reference_rows,
    }
    payload["frozen_external_digest"]=hashlib.sha256(json.dumps({"archive_lock_digest":payload["archive_lock_digest"],"parent_v60_digest":payload["parent_v60_digest"],"protocol":payload["protocol"],"dataset_summaries":summaries,"state_input_sha256":payload["state_input_sha256"],"state_digests":[row["state_digest"] for row in reference_rows]},sort_keys=True).encode("utf-8")).hexdigest()
    reference_path.parent.mkdir(parents=True,exist_ok=True)
    reference_path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--states",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.states,args.reference)
    print(json.dumps({"status":result["status"],"datasets":result["contributing_dataset_count"],"base_states":result["base_state_count"],"profiled_states":result["profiled_state_count"],"pareto_solved":result["pareto_solved_count"],"plain_solved":result["both_solved_count"],"pareto_only":result["pareto_only_count"]},indent=2))


if __name__=="__main__":
    main()
