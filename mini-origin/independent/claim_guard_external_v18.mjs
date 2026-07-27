import crypto from "node:crypto";
import fs from "node:fs";
import { spawnSync } from "node:child_process";

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

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function baseContract() {
  return {
    claim_id: "mini-origin-claim-guard-v18",
    candidate_hash: sha("external-frozen-candidate-v18"),
    required_runs: 5,
    min_successes: 4,
    score_threshold: 0.78,
    median_threshold: 0.82,
    min_control_gap: 0.08,
    median_control_gap: 0.10,
    min_ablation_gap: 0.12,
    oracle_ceiling: 1.0,
    operation_budget: 0.20,
    required_checks: [
      "fixed_boundary",
      "bipolar",
      "sealed_holdout",
      "shortcut_resistance",
      "control_present",
    ],
  };
}

function buildValid(contract, seed) {
  const contractDigest = sha(stable(contract));
  const seeds = Array.from({ length: contract.required_runs }, (_, index) => seed * 1000 + index + 1);
  const manifest = {
    contract_digest: contractDigest,
    candidate_hash: contract.candidate_hash,
    seeds,
    commitment: sha(`${contractDigest}:sealed-v1:${seeds.join(",")}`),
    issued_after_freeze: true,
  };
  const manifestDigest = sha(stable(manifest));
  const checks = contract.required_checks.map((name) => [name, true]);
  const runs = seeds.map((hiddenSeed, index) => {
    const score = 0.842 + 0.004 * Math.sin(seed + index);
    return {
      seed: hiddenSeed,
      score,
      control: score - 0.125,
      ablation: score - 0.165,
      candidate_budget: 0.20,
      control_budget: 0.20,
      threshold_used: contract.score_threshold,
      contract_digest: contractDigest,
      candidate_hash: contract.candidate_hash,
      manifest_digest: manifestDigest,
      holdout_candidates: 1,
      selected_after_holdout: false,
      holdout_policy_violations: 0,
      checks,
    };
  });
  return {
    contract,
    manifest,
    bundle: {
      claim_id: contract.claim_id,
      claimed_breakthrough: true,
      runs,
    },
  };
}

function refreshManifest(caseValue) {
  const digest = sha(stable(caseValue.contract));
  caseValue.manifest.contract_digest = digest;
  caseValue.manifest.candidate_hash = caseValue.contract.candidate_hash;
  caseValue.manifest.commitment = sha(`${digest}:sealed-v1:${caseValue.manifest.seeds.join(",")}`);
  const manifestDigest = sha(stable(caseValue.manifest));
  caseValue.bundle.claim_id = caseValue.contract.claim_id;
  for (const run of caseValue.bundle.runs) {
    run.contract_digest = digest;
    run.candidate_hash = caseValue.contract.candidate_hash;
    run.manifest_digest = manifestDigest;
    run.threshold_used = caseValue.contract.score_threshold;
  }
}

function caseOf(name, expectedAccept, value) {
  return { name, expected_accept: expectedAccept, ...value };
}

function intrinsic(seed, name, mutateContract, mutateEvidence = null) {
  const contract = baseContract();
  mutateContract(contract);
  const value = buildValid(contract, seed);
  if (mutateEvidence) mutateEvidence(value);
  refreshManifest(value);
  return caseOf(name, false, value);
}

function externalCases(seed) {
  const valid = buildValid(baseContract(), seed);
  const cases = [caseOf("independent_valid", true, clone(valid))];

  let value = clone(valid);
  value.bundle.runs[0].candidate_budget = -0.20;
  value.bundle.runs[0].control_budget = -0.20;
  cases.push(caseOf("negative_equal_budgets", false, value));

  value = clone(valid);
  value.bundle.runs[0].control = -0.10;
  cases.push(caseOf("negative_control_metric", false, value));

  value = clone(valid);
  value.bundle.runs[0].ablation = -0.10;
  cases.push(caseOf("negative_ablation_metric", false, value));

  value = clone(valid);
  value.bundle.runs[0].holdout_candidates = 0;
  cases.push(caseOf("zero_holdout_candidates", false, value));

  value = clone(valid);
  value.bundle.runs[0].holdout_candidates = -2;
  cases.push(caseOf("negative_holdout_candidates", false, value));

  value = clone(valid);
  value.manifest.seeds[0] = 0;
  value.bundle.runs[0].seed = 0;
  refreshManifest(value);
  cases.push(caseOf("nonpositive_hidden_seed", false, value));

  cases.push(intrinsic(seed, "zero_min_successes", (contract) => { contract.min_successes = 0; }));
  cases.push(intrinsic(seed, "negative_score_threshold", (contract) => { contract.score_threshold = -0.2; }));
  cases.push(intrinsic(seed, "negative_median_threshold", (contract) => { contract.median_threshold = -0.2; }));
  cases.push(intrinsic(seed, "negative_control_gap", (contract) => { contract.min_control_gap = -0.1; contract.median_control_gap = -0.1; }));
  cases.push(intrinsic(seed, "negative_ablation_gap", (contract) => { contract.min_ablation_gap = -0.1; }));
  cases.push(intrinsic(seed, "empty_required_checks", (contract) => { contract.required_checks = []; }));
  cases.push(intrinsic(seed, "empty_candidate_hash", (contract) => { contract.candidate_hash = ""; }));
  cases.push(intrinsic(seed, "empty_claim_id", (contract) => { contract.claim_id = ""; }));
  cases.push(intrinsic(
    seed,
    "negative_operation_budget",
    (contract) => { contract.operation_budget = -0.2; },
    (entry) => {
      for (const run of entry.bundle.runs) {
        run.candidate_budget = -0.2;
        run.control_budget = -0.2;
      }
    },
  ));

  value = clone(valid);
  value.bundle.claimed_breakthrough = false;
  cases.push(caseOf("valid_evidence_without_claim", false, value));

  value = clone(valid);
  value.bundle.runs[0].checks = value.bundle.runs[0].checks.filter(([name]) => name !== "control_present");
  cases.push(caseOf("missing_required_check", false, value));

  return cases;
}

function main() {
  const args = process.argv.slice(2);
  const seedIndex = args.indexOf("--seed");
  const outputIndex = args.indexOf("--output");
  const seed = seedIndex >= 0 ? Number(args[seedIndex + 1]) : 301;
  const output = outputIndex >= 0 ? args[outputIndex + 1] : "results/claim-guard-external.json";
  const casesPath = `${output}.cases.json`;
  const adapterPath = `${output}.adapter.json`;
  fs.mkdirSync(output.substring(0, output.lastIndexOf("/")), { recursive: true });
  fs.writeFileSync(casesPath, JSON.stringify({ cases: externalCases(seed) }, null, 2));

  const run = spawnSync(
    "python",
    ["-m", "mini_origin.claim_guard_adapter_v18", "--input", casesPath, "--output", adapterPath],
    { encoding: "utf8" },
  );
  if (run.status !== 0) {
    console.error(run.stdout);
    console.error(run.stderr);
    process.exit(run.status ?? 1);
  }
  const report = JSON.parse(fs.readFileSync(adapterPath, "utf8"));
  const failures = report.results.filter((item) => !item.correct);
  const summary = {
    seed,
    independent_language: "JavaScript",
    mutation_suite_shared_with_python: false,
    case_count: report.case_count,
    correct_count: report.correct_count,
    accuracy: report.accuracy,
    all_correct: report.all_correct,
    missed_cases: failures.map((item) => item.name),
    results: report.results,
  };
  fs.writeFileSync(output, JSON.stringify(summary, null, 2));
  console.log(JSON.stringify({
    seed,
    accuracy: summary.accuracy,
    all_correct: summary.all_correct,
    missed_cases: summary.missed_cases,
  }, null, 2));
}

main();
