# V2 Corrective Experiment Record

## Status

**Completed and rejected.** V1 remains the selected release.

## Goal

Recover V1's weak clarification behavior and reduce unnecessary tool invocation without materially sacrificing its valid structured-call performance.

## Intervention

V2 continued training from the V1 adapter at a small corrective learning rate using 600 examples:

- 300 missing-required-argument clarification examples
- 100 noisy `get_order` examples
- 100 complete normal tool calls
- 100 conceptual/no-tool examples

The locked 150-case final benchmark was unchanged for the Base/V1/V2 comparison.

## Promotion criteria

- Tool selection accuracy at least 90%
- Strict exact-call accuracy at least 80%
- Positive clarification recovery versus V1
- Lower hallucinated-tool rate than V1

These were decision thresholds, not result claims.

## Measured result

| Metric | V2 |
|---|---:|
| Tool selection accuracy | 44.44% |
| Strict exact-call accuracy | 42.22% |
| Argument KV accuracy | 43.70% |
| Tool parse validity | 44.44% |
| Strict behavior failures | 64 / 150 |
| Overall strict behavior accuracy | 57.33% |

V2 became too conservative and frequently requested information already supplied in otherwise valid requests.

## Decision

V2 failed the promotion criteria and was rejected. No additional training was performed on the locked final benchmark. A future V3, if pursued, should use a fresh development set with matched contrast pairs rather than treating the final benchmark as training feedback.
