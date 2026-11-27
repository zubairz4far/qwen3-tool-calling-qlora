# V3 — reproducible behavioral post-training experiment

Status: implementation candidate; no model has been trained or promoted by this change.
V1/V2 historical evidence and published adapters remain unchanged.

## Contract and limitations
The seven tool names follow the historical README, but argument definitions are a
**proposed V3 contract**, not reconstructed original V1 schemas. The original V1
train/validation and locked 150-case files are not in the inspected repository.
When recovered, archive them with their original hashes and compare schemas before
attempting an exact historical reproduction. All V3 examples are synthetic; callback
identifiers are non-dialable, and no tool executes.

The canonical schema and prompt are shared by V3 data generation, training and
evaluation. Historical V2 and demo code remain untouched for provenance.
The deterministic generator yields 1,260 training / 420 development rows across
complete calls, missing information, conceptual requests, multi-turn completion
and conflicting instructions. Train/dev template families and synthetic identifiers
are separated; some closed-set statuses necessarily overlap. Exact duplicates and
cross-split conversation overlap fail validation. Template-based diversity does NOT
establish semantic independence or real-world generalization. The development set
is not a final holdout. Independently author and freeze a human-reviewed final suite
before any release comparison.

## CPU validation
    python src/v3_data.py --out outputs/v3_data
    python -m unittest discover -s tests -v

## GPU experiment
Use one visible NVIDIA GPU; install the project requirements in an isolated
environment. Pin and record the compatible package versions before comparative
runs. Resolve full revision SHAs from Hugging Face for the base model and V1 adapter.

    CUDA_VISIBLE_DEVICES=0 python src/train_v3.py --data outputs/v3_data \
      --output outputs/v3_candidate --base-revision BASE_SHA --adapter-revision V1_SHA

    CUDA_VISIBLE_DEVICES=0 python src/evaluate_v3.py --data outputs/v3_data \
      --base-revision BASE_SHA --adapter zubairz4far/qwen3-1.7b-tool-calling \
      --adapter-revision V1_SHA --output outputs/v3_eval_v1

    CUDA_VISIBLE_DEVICES=0 python src/evaluate_v3.py --data outputs/v3_data \
      --base-revision BASE_SHA --adapter outputs/v3_candidate --output outputs/v3_eval_candidate

Run the same evaluator with no adapter for the base baseline. The training script
records package versions, dataset manifest hash, revisions, config and metrics;
evaluation retains raw outputs. Exact reference text is deliberately conservative
for clarification and direct answers; review paraphrases with a blinded rubric.
No automatic deployment, Hub upload, or claims of accuracy.

## Proposed promotion protocol
Freeze a new independently written final dataset and scoring rubric before tuning.
Compare base, V1 and V3 at identical decoding settings; retain model-only versus
guarded system metrics separately. Suggested minimums: >=90% valid exact calls,
>=80% correct clarification, >=80% useful direct answers (human rubric), zero
unauthorized executed actions, and no meaningful loss on valid calls versus V1.
Unexpected attempted calls must be reported separately from executed actions.
Run at least three seeds and report variability before claiming improvement.
If V3 fails, retain V1 as historical release and keep deterministic gateway protections.
Do not train on a locked benchmark. GRPO is a later experiment, not part of this change.
