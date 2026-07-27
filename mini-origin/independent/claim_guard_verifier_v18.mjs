import crypto from "node:crypto";
import fs from "node:fs";

const FLOAT_FIELDS = new Set([
  "score_threshold",
  "median_threshold",
  "min_control_gap",
  "median_control_gap",
  "min_ablation_gap",
  "oracle_ceiling",
  "operation_budget",
]);
const SHA256 = /^[0-9a-f]{64}$/;

function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function sha(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function canonicalFloat(value) {
  if (!Number.isFinite(value)) throw new Error("non-finite contract number");
  if (Object.is(value, -0) || value === 0) return "0";
  let text = Number(value).toPrecision(17);
  const parts = text.split("e");
  let mantissa = parts[0];
  if (mantissa.includes(".")) mantissa = mantissa.replace(/0+$/, "").replace(/\.$/, "");
  if (parts.length === 1) return mantissa;
  const sign = parts[1].startsWith("-") ? "-" : "+";
  const exponent = parts[1].replace(/^[+-]/, "").replace(/^0+(?=\d)/, "");
  return `${mantissa}e${sign}${exponent}`;
}

function contractDigest(contract) {
  const payload = JSON.parse(JSON.stringify(contract));
  for (const field of FLOAT_FIELDS) {
    payload[field] = `float:${canonicalFloat(Number(payload[field]))}`;
  }
  return sha(stable(payload));
}

function manifestDigest(manifest) {
  return sha(stable(manifest));
}

function median(values) {
  if (!values.length) return Number.NEGATIVE_INFINITY;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function unique(values) {
  return new Set(values).size === values.length;
}

function finite(...values) {
  return values.every(Number.isFinite);
}

function verify(contract, manifest, bundle) {
  const failures = [];
  const digest = contractDigest(contract);

  if (typeof contract.claim_id !== "string" || contract.claim_id.trim() === "") failures.push("contract_identity");
  if (typeof contract.candidate_hash !== "string" || !SHA256.test(contract.candidate_hash)) failures.push("candidate_hash");
  if (!Number.isInteger(contract.required_runs) || contract.required_runs <= 0) failures.push("run_count");
  if (!Number.isInteger(contract.min_successes) || contract.min_successes < 1 || contract.min_successes > contract.required_runs) failures.push("success_requirement");
  if (!Array.isArray(contract.required_checks) || !contract.required_checks.length) failures.push("required_checks");
  if (Array.isArray(contract.required_checks)) {
    if (!contract.required_checks.every((name) => typeof name === "string" && name.trim() !== "")) failures.push("required_check_name");
    if (!unique(contract.required_checks)) failures.push("duplicate_required_check");
  }
  const contractNumbers = [
    contract.score_threshold,
    contract.median_threshold,
    contract.min_control_gap,
    contract.median_control_gap,
    contract.min_ablation_gap,
    contract.oracle_ceiling,
    contract.operation_budget,
  ];
  if (!finite(...contractNumbers)) failures.push("contract_number");
  if (!(contract.oracle_ceiling > 0)) failures.push("oracle");
  if (!(contract.score_threshold >= 0 && contract.score_threshold <= contract.oracle_ceiling)) failures.push("score_threshold");
  if (!(contract.median_threshold >= 0 && contract.median_threshold <= contract.oracle_ceiling)) failures.push("median_threshold");
  if (!(contract.min_control_gap >= 0 && contract.median_control_gap >= 0)) failures.push("control_gap");
  if (!(contract.min_ablation_gap >= 0)) failures.push("ablation_gap");
  if (!(contract.operation_budget >= 0)) failures.push("operation_budget");

  if (manifest.contract_digest !== digest) failures.push("manifest_contract");
  if (manifest.candidate_hash !== contract.candidate_hash) failures.push("manifest_candidate");
  if (manifest.issued_after_freeze !== true) failures.push("freeze_order");
  if (!Array.isArray(manifest.seeds) || manifest.seeds.length !== contract.required_runs) failures.push("manifest_seed_count");
  if (!Array.isArray(manifest.seeds) || !unique(manifest.seeds)) failures.push("manifest_seed_unique");
  if (!Array.isArray(manifest.seeds) || !manifest.seeds.every((seed) => Number.isInteger(seed) && seed > 0)) failures.push("manifest_seed_value");
  const commitment = sha(`${digest}:sealed-v1:${(manifest.seeds ?? []).join(",")}`);
  if (manifest.commitment !== commitment) failures.push("commitment");

  if (bundle.claim_id !== contract.claim_id) failures.push("bundle_identity");
  if (!Array.isArray(bundle.runs) || bundle.runs.length !== contract.required_runs) failures.push("bundle_run_count");
  const runSeeds = Array.isArray(bundle.runs) ? bundle.runs.map((run) => run.seed) : [];
  if (!unique(runSeeds)) failures.push("run_seed_unique");
  if (!runSeeds.every((seed) => Number.isInteger(seed) && seed > 0)) failures.push("run_seed_value");
  if (stable([...runSeeds].sort((a, b) => a - b)) !== stable([...(manifest.seeds ?? [])].sort((a, b) => a - b))) failures.push("seed_set");

  const scores = [];
  const controlGaps = [];
  const ablationGaps = [];
  let successes = 0;
  const sealedDigest = manifestDigest(manifest);

  for (const [index, run] of (bundle.runs ?? []).entries()) {
    const checks = Array.isArray(run.checks) ? run.checks : [];
    const names = checks.map((entry) => entry[0]);
    const checkMap = new Map(checks);
    if (!unique(names)) failures.push(`run_${index}_duplicate_check`);
    for (const required of contract.required_checks ?? []) {
      if (checkMap.get(required) !== true) failures.push(`run_${index}_check_${required}`);
    }
    if (run.contract_digest !== digest) failures.push(`run_${index}_contract`);
    if (run.candidate_hash !== contract.candidate_hash) failures.push(`run_${index}_candidate`);
    if (run.manifest_digest !== sealedDigest) failures.push(`run_${index}_manifest`);
    if (!Number.isFinite(run.threshold_used) || Math.abs(run.threshold_used - contract.score_threshold) > 1e-12) failures.push(`run_${index}_threshold`);
    if (!Number.isInteger(run.holdout_candidates) || run.holdout_candidates !== 1 || run.selected_after_holdout !== false) failures.push(`run_${index}_holdout_selection`);
    if (!Number.isInteger(run.holdout_policy_violations) || run.holdout_policy_violations !== 0) failures.push(`run_${index}_holdout_policy`);

    const metrics = [run.score, run.control, run.ablation, run.candidate_budget, run.control_budget];
    if (!finite(...metrics)) {
      failures.push(`run_${index}_metric`);
      continue;
    }
    if (!(run.score >= 0 && run.score <= contract.oracle_ceiling)) failures.push(`run_${index}_score_range`);
    if (!(run.control >= 0 && run.control <= contract.oracle_ceiling)) failures.push(`run_${index}_control_range`);
    if (!(run.ablation >= 0 && run.ablation <= contract.oracle_ceiling)) failures.push(`run_${index}_ablation_range`);
    if (!(run.candidate_budget >= 0 && run.control_budget >= 0)) failures.push(`run_${index}_budget_sign`);
    if (run.candidate_budget > contract.operation_budget + 1e-12) failures.push(`run_${index}_candidate_budget`);
    if (run.control_budget > contract.operation_budget + 1e-12) failures.push(`run_${index}_control_budget`);
    if (Math.abs(run.candidate_budget - run.control_budget) > 1e-12) failures.push(`run_${index}_budget_parity`);

    const controlGap = run.score - run.control;
    const ablationGap = run.score - run.ablation;
    if (controlGap < contract.min_control_gap) failures.push(`run_${index}_control_gap`);
    if (ablationGap < contract.min_ablation_gap) failures.push(`run_${index}_ablation_gap`);
    scores.push(run.score);
    controlGaps.push(controlGap);
    ablationGaps.push(ablationGap);
    if (run.score >= contract.score_threshold && controlGap >= contract.min_control_gap && ablationGap >= contract.min_ablation_gap) successes += 1;
  }

  if (successes < contract.min_successes) failures.push("aggregate_successes");
  if (median(scores) < contract.median_threshold) failures.push("aggregate_score");
  if (median(controlGaps) < contract.median_control_gap) failures.push("aggregate_control");
  if ((ablationGaps.length ? Math.min(...ablationGaps) : Number.NEGATIVE_INFINITY) < contract.min_ablation_gap) failures.push("aggregate_ablation");

  return {
    accepted: failures.length === 0 && bundle.claimed_breakthrough === true,
    failures: [...new Set(failures)],
    successful_runs: successes,
  };
}

function main() {
  const args = process.argv.slice(2);
  const inputIndex = args.indexOf("--input");
  const outputIndex = args.indexOf("--output");
  if (inputIndex < 0 || outputIndex < 0) throw new Error("--input and --output are required");
  const input = JSON.parse(fs.readFileSync(args[inputIndex + 1], "utf8"));
  const results = input.cases.map((testCase) => {
    const verdict = verify(testCase.contract, testCase.manifest, testCase.bundle);
    return {
      name: testCase.name,
      expected_accept: Boolean(testCase.expected_accept),
      observed_accept: verdict.accepted,
      correct: verdict.accepted === Boolean(testCase.expected_accept),
      failures: verdict.failures,
    };
  });
  const report = {
    implementation: "independent-javascript-verifier",
    case_count: results.length,
    correct_count: results.filter((result) => result.correct).length,
    accuracy: results.filter((result) => result.correct).length / Math.max(1, results.length),
    all_correct: results.every((result) => result.correct),
    results,
  };
  fs.writeFileSync(args[outputIndex + 1], JSON.stringify(report, null, 2));
  console.log(JSON.stringify({
    case_count: report.case_count,
    accuracy: report.accuracy,
    all_correct: report.all_correct,
    missed: results.filter((result) => !result.correct).map((result) => result.name),
  }, null, 2));
}

main();
