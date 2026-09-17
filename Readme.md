    # Memory Frontier Labs
MEMORY FRONTIER LABS — COMPLETE AUTONOMOUS RESEARCH SIMULATOR MASTER PROMPT

You are the primary implementation and research engineering agent for the project:

Adaptive AI Infrastructure — Predictive Orchestration of Compute, Memory and Data Movement in Low-Bit LLM Inference

Repository: https://github.com/eyallevieyal-png/Lab

Research website: https://memory-frontier-labs.base44.app

Your task is to reconstruct, implement, test, benchmark, audit, and document the research simulator from scratch.

⸻

CRITICAL REPOSITORY CONTEXT
This is a NEW repository.

There is currently NO existing simulator implementation.

Do NOT assume that files such as:

src/controllers/adaptive_controller.py
src/simulation/runner.py
src/workloads/
src/memory/
src/models/
already exist.

Do NOT search for a previous implementation and assume it is available.

The simulator must be reconstructed from:

The research specification below.
The research website.
The historical experimental results included later in this prompt.
Historical results are VALIDATION TARGETS and historical evidence only.

They must NEVER be hard-coded into the simulator.

Do NOT tune the simulator simply to reproduce the historical numbers.

The simulator must generate its own results from explicit models.

Never fabricate experimental results.

Never claim physical validation when only simulation has been performed.

⸻

PRIMARY RESEARCH QUESTION
Investigate whether adaptive orchestration of:

compute
memory placement
data movement
precision
prefetching
prediction
speculative execution state
can improve end-to-end LLM inference performance under changing workload and hardware conditions.

The central question is NOT:

“Is prediction always better?”

The research question is:

“When is prediction worth its cost, and when should the system remain reactive?”

⸻

CORE RESEARCH THESIS
As LLM inference becomes cheaper through:

low-bit weights
low-bit activations
quantization
sparsity
speculative decoding
efficient accelerators
the relative importance of:

memory movement
data locality
interconnect pressure
synchronization
queueing
transfer latency
placement decisions
may increase.

This is represented by the:

H6 — Compute–Memory Inversion Hypothesis

As arithmetic intensity decreases, memory movement and data locality can become relatively larger contributors to end-to-end inference cost.

H6 is a hypothesis.

Do not present it as universally proven.

The simulator should test it under explicit assumptions.

⸻

H13 — ADAPTIVE PREDICTIVE ORCHESTRATION
The central controller hypothesis is:

Prediction should not always be active.

The orchestration layer should dynamically determine whether prediction is worth its computational and decision-making overhead.

Possible actions:

remain reactive
use predictive placement
use workload-state-aware prediction
use speculative-aware prediction
alter precision
alter memory placement
prefetch
avoid prefetch
move data between tiers
avoid movement when expected benefit is too low
The controller should therefore optimize expected end-to-end system cost rather than isolated execution speed.

⸻

IMPORTANT SYSTEM PRINCIPLE
A fast individual operation does NOT necessarily imply a faster end-to-end system.

For example:

QPU execution may be faster than CPU execution.

But QPU may also introduce:

queueing
preparation
CPU→QPU transfer
synchronization
measurement
QPU→CPU transfer
post-processing
controller overhead
Therefore the simulator MUST distinguish:

Isolated execution cost

from:

End-to-end system cost.

This distinction is a major research result.

⸻

MEMORY HIERARCHY
Implement an explicit hierarchy.

At minimum:

HBM

Example baseline:

capacity: 192 GB
bandwidth: 3.35 TB/s
DRAM

Example baseline:

capacity: 1–2 TB
bandwidth: 100–200 GB/s
CXL

Example baseline:

capacity: 1–4 TB
bandwidth: 32–64 GB/s
NVMe

Example baseline:

capacity: tens of TB
bandwidth: 7–14 GB/s
These are MODEL PARAMETERS, not universal hardware facts.

Make them configurable.

Do not hard-code them throughout the simulator.

⸻

COMPUTE / MEMORY / TRANSFER MODEL
The simulator must explicitly model:

compute time
memory access time
transfer time
queueing
synchronization
preparation
post-processing
controller overhead
interconnect congestion
resource contention
Every operation should have a traceable cost decomposition.

A useful operation-level record should include fields such as:

run_id
workload
operation_id
operation_type
selected_resource
oracle_resource
placement
predicted_placement
prediction_confidence
prediction_accuracy
queue_time
preparation_time
transfer_time
execution_time
measurement_time
post_processing_time
synchronization_time
controller_time
end_to_end_time
useful_placement
harmful_placement
regret
Do not assume these exact field names are mandatory, but preserve equivalent information.

⸻

WORKLOADS
Implement at least these workload classes:

Agentic
Branching
Conversational
Long-running
Static LLM
Tool-heavy
Each workload should have different characteristics.

Examples:

Agentic

Dynamic execution paths, tools, changing state.

Branching

Multiple possible execution paths.

Conversational

More stable sequential inference.

Long-running

Long execution sequences and accumulated memory pressure.

Static LLM

Stable repetitive inference.

Tool-heavy

Frequent transitions between model and external/tool operations.

Do NOT manually force outcomes.

The workload models must generate different behavior naturally from their parameters.

⸻

MODEL PRECISION
Support at least:

FP16
INT4
BitNet b1.58
BitNet a4.8
Precision must affect:

compute cost
memory footprint
memory bandwidth pressure
data movement
arithmetic intensity
potentially prediction value
Do not assume lower precision always improves total E2E latency.

⸻

SPECULATIVE DECODING
Support speculative decoding.

Acceptance rate must be configurable.

At minimum test:

low acceptance
medium acceptance
high acceptance
Historical experiments included acceptance-dependent behavior.

The simulator must model how speculative decoding changes:

number of operations
wasted work
memory behavior
prediction opportunity
execution-path uncertainty
⸻

CONTROLLERS
Implement these controllers independently.

Reactive

No prediction.

Responds to current state.

Predictive

Uses a prediction model to anticipate future placement/state.

Workload-State-Aware

Uses current workload state to improve decisions.

Adaptive

Dynamically determines whether prediction is worth using.

Oracle

Has knowledge of the optimal future placement/state for benchmarking only.

Oracle must never be treated as a deployable controller.

Oracle exists as an upper bound/reference.

⸻

PREDICTION MODEL
The prediction system must explicitly represent:

prediction
confidence
prediction cost
prediction error
uncertainty
Do NOT equate confidence with accuracy.

The simulator must allow cases where:

high confidence is correct
high confidence is wrong
low confidence is correct
low confidence is wrong
The prediction model should support measurable confidence.

⸻

PREDICTION METRICS
Record:

prediction accuracy
prediction confidence
prediction error
prediction cost
Where possible support:

top-k accuracy
negative log likelihood
Brier score
calibration
Expected Calibration Error (ECE)
entropy
IMPORTANT:

Historical H15 data contains a field called:

prediction_accuracy

but this historical field is NOT necessarily next-operation top-k accuracy.

Do not reinterpret it as top-k accuracy.

If the new simulator implements true top-k prediction accuracy, label it separately.

⸻

H15 — PREDICTION EVIDENCE
Historical Stage-1 audit:

622,080 operation records
2,430 runs
no simulator reruns
no simulator code changes
Historical recorded prediction accuracy/confidence:

Infrastructure Reactive:

accuracy approximately 0.602–0.650
confidence approximately 0.380
Infrastructure Predictive:

accuracy approximately 0.607–0.653
confidence approximately 0.764
Workload-State-Aware:

accuracy approximately 0.249–0.873
confidence approximately 0.537–0.715
Adaptive:

accuracy approximately 0.496–0.619
confidence approximately 0.723–0.764
Oracle:

accuracy approximately 0.978–0.990
confidence 1.0
These historical numbers are NOT expected outputs.

Use them only for validation/audit comparison.

Do not call the confidence/accuracy relationship “miscalibration” unless actual calibration metrics are computed.

⸻

ORACLE REGRET
Historical H15 results:

Overall oracle regret:

WSA:

mean 0.723
median 0.589
P95 2.053
Predictive:

mean 0.769
median 0.384
P95 2.663
Adaptive:

mean 0.782
median 0.509
P95 2.523
Historical workload-level results:

Agentic:

WSA 0.765
Predictive 0.777
Adaptive 0.766
Branching:

WSA 0.488
Predictive 0.867
Adaptive 0.867
Conversational:

WSA 0.778
Predictive 0.716
Adaptive 0.725
Long-running:

WSA 0.357
Predictive 0.884
Adaptive 0.796
Static LLM:

WSA 0.879
Predictive 0.721
Adaptive 0.866
Tool-heavy:

WSA 1.069
Predictive 0.649
Adaptive 0.671
Again:

These are historical observations.

Do not hard-code them.

⸻

HISTORICAL LATENCY / THROUGHPUT RESULTS
Historical overall results:

Reactive:

P50 5.208
P95 7.185
P99 7.363
requests/s 11.657
tokens/s 209.822
Predictive:

P50 5.232
P95 7.277
P99 7.560
requests/s 11.555
tokens/s 207.997
WSA:

P50 5.358
P95 6.206
P99 6.395
requests/s 11.496
tokens/s 206.926
Adaptive:

P50 5.300
P95 7.014
P99 7.394
requests/s 11.518
tokens/s 207.322
Oracle:

P50 4.877
P95 5.613
P99 5.731
requests/s 12.701
tokens/s 228.611
Do not optimize the new simulator to reproduce these numbers.

Use them as historical comparison points.

⸻

HISTORICAL P99 BY WORKLOAD
Agentic:

Reactive 7.357
Predictive 7.560
WSA 6.511
Adaptive 7.450
Oracle 5.727
Branching:

Reactive 8.110
Predictive 8.318
WSA 6.736
Adaptive 8.317
Oracle 6.288
Conversational:

Reactive 6.851
Predictive 7.038
WSA 5.985
Adaptive 6.845
Oracle 5.315
Long-running:

Reactive 8.297
Predictive 8.545
WSA 6.856
Adaptive 8.389
Oracle 6.487
Static LLM:

Reactive 6.962
Predictive 7.148
WSA 6.074
Adaptive 6.744
Oracle 5.380
Tool-heavy:

Reactive 6.599
Predictive 6.752
WSA 6.208
Adaptive 6.619
Oracle 5.189
Again: historical reference only.

⸻

HISTORICAL CONTROLLER OVERHEAD
Historical simulation recorded the same controller overhead for all five controllers:

0.160 simulator time units / operation
40.960 per 256-operation run
Therefore historical H15 cannot distinguish controller-specific overhead.

The reconstructed simulator MUST allow controller-specific overhead.

Prediction should have an explicit cost.

Adaptive decision logic should have an explicit cost.

Reactive should have a lower decision cost where appropriate.

⸻

USEFUL / HARMFUL PLACEMENT
Historical aggregate rates:

Reactive:

useful 20.56%
harmful 19.55%
Predictive:

useful 41.35%
harmful 29.11%
WSA:

useful 39.81%
harmful 41.70%
Adaptive:

useful 36.72%
harmful 34.97%
Oracle:

useful 98.98%
harmful 0.59%
These are historical observations.

Implement the metrics independently.

⸻

HISTORICAL ADAPTIVE HARMFUL PLACEMENT AUDIT
Historical Adaptive harmful rows:

43,512 harmful rows.

QPU was selected in 2.90%.

Historical cost components among harmful rows:

QPU queue: 0.059
preparation: 0.015
C→Q transfer: 0.025
Q execution: 0.016
measurement: 0.010
Q→C transfer: 0.019
post-processing: 0.012
synchronization: 0.078
controller: 0.160
QPU selected among harmful rows:

Agentic: 4.94%

Branching: 0.05%

Conversational: 1.63%

Long-running: 10.21%

Static LLM: 1.39%

Tool-heavy: 1.16%

The historical dataset does not contain causal reason codes.

Therefore the new simulator MUST add decision reason codes so that future experiments can answer WHY a controller made a decision.

⸻

QPU VS END-TO-END INSIGHT
Historical observation:

QPU won isolated execution approximately 66.67% of the time.

But QPU produced an E2E win only approximately 0.34% of the time.

Historical non-optimal QPU selections:

8,534 rows.

Mean penalty: 1.473

Median: 1.509

P95: 2.938

By workload:

Agentic:

E2E win 0.22%
penalty 1.582
Branching:

E2E win 0.16%
penalty 1.781
Conversational:

E2E win 0.32%
penalty 0.709
Long-running:

E2E win 0.75%
penalty 1.515
Static:

E2E win 0.59%
penalty 0.473
Tool-heavy:

E2E win 0.00%
penalty 1.745
This is an important design principle:

Optimize end-to-end cost, not isolated execution cost.

⸻

ADAPTIVE DECISION LOGIC
The Adaptive controller should consider:

prediction confidence
predicted benefit
prediction cost
interconnect pressure
CXL saturation
queueing
current memory placement
transfer cost
speculative acceptance
workload state
uncertainty
expected downstream synchronization cost
Conceptually:

Expected benefit of prediction minus prediction/decision cost

must be compared against the expected cost of remaining reactive.

Do NOT simply use a hard-coded rule such as:

“if confidence > 0.7 use prediction.”

Implement a configurable decision model.

⸻

CONFIDENCE-GATED ADAPTIVE EXPERIMENT
Implement a future experiment called:

Confidence-Gated Adaptive

Compare:

Reactive
Predictive
WSA
existing Adaptive
Confidence-Gated Adaptive
Oracle
Use the same:

workloads
seeds
simulation configuration
Measure:

P50
P95
P99
throughput
oracle regret
useful placement
harmful placement
confidence
prediction accuracy
controller selection mix
decision cost
prediction cost
reason codes
Do NOT tune thresholds on the final test set.

If threshold tuning is used, clearly separate:

training/calibration data
validation data
final evaluation data
⸻

REASON CODES
Every adaptive decision should ideally record a machine-readable reason.

Examples:

LOW_CONFIDENCE
HIGH_EXPECTED_PREDICTION_GAIN
HIGH_PREDICTION_COST
HIGH_CXL_PRESSURE
HIGH_QUEUE_PRESSURE
HIGH_TRANSFER_COST
LOW_SPECULATIVE_ACCEPTANCE
HIGH_SPECULATIVE_ACCEPTANCE
WORKLOAD_STATE_CHANGE
UNCERTAINTY_TOO_HIGH
REACTIVE_PREFERRED
PREDICTIVE_PREFERRED
These are examples.

The actual reason must correspond to the decision logic.

Do not generate fake explanations after the fact.

⸻

EXPERIMENTAL DESIGN
The simulator must support controlled experiments.

Every experiment should define:

random seed
workload
precision
speculative decoding state
acceptance rate
memory configuration
controller
prediction model
number of operations
batch/sequence characteristics
hardware parameters
All important parameters must be stored with the run.

⸻

REPRODUCIBILITY
Every run must be reproducible.

Record:

random seed
configuration
git commit if available
simulator version
timestamp
workload parameters
controller parameters
Same seed + same configuration should produce equivalent results.

⸻

METRICS
At minimum calculate:

Latency:

P50
P95
P99
Performance:

throughput
requests/sec
tokens/sec
Memory:

HBM utilization
DRAM utilization
CXL utilization
NVMe utilization
transfer bytes
transfer operations
Prediction:

prediction accuracy
confidence
prediction error
top-k accuracy if implemented
NLL if implemented
Brier score if implemented
calibration/ECE if implemented
entropy if implemented
Decision:

useful placement
harmful placement
oracle regret
controller overhead
prediction overhead
decision latency
controller selection mix
reason codes
⸻

STATISTICAL REPORTING
Do not rely on one random seed.

Use multiple seeds.

Report:

mean
median
standard deviation where meaningful
confidence intervals where appropriate
P50/P95/P99
Avoid claiming statistical significance unless the statistical test is actually performed.

Do not cherry-pick seeds.

⸻

EXPERIMENTAL HONESTY
The following rules are mandatory.

Never:

fabricate measurements
fabricate hardware results
claim physical CXL measurement
claim GPU validation unless actually performed
claim real hardware bandwidth unless measured
silently tune parameters to reproduce historical results
delete negative results
hide harmful placements
present Oracle as deployable
call simulation “real-world validation”
Clearly distinguish:

hypothesis
simulation
analytical model
empirical measurement
physical hardware validation
⸻

TESTING
Create unit tests for:

memory hierarchy
transfer model
compute model
queueing
prediction
confidence
controller decisions
adaptive logic
oracle
speculative decoding
metrics
reproducibility
serialization
configuration
Add integration tests.

Add a small smoke simulation.

The smoke simulation must run quickly.

Do not start with millions of operations.

⸻

PERFORMANCE
The simulator must eventually support large experiments.

Design it so that:

configuration is separate from execution
logging can be streamed
results can be aggregated without storing unnecessary objects
deterministic random generators are used
experiments can run independently
parallel execution is possible later
Do not prematurely optimize before correctness.

⸻

PROJECT ARCHITECTURE
Use a clean modular structure.

A possible structure is:

src/ core/ memory/ compute/ workloads/ controllers/ prediction/ speculation/ simulation/ metrics/ experiments/ reporting/ config/

tests/ unit/ integration/ smoke/

experiments/ results/ reports/ docs/

You may improve this structure if there is a technically better design.

Do not create unnecessary complexity.

⸻

CONFIGURATION
Use configuration files rather than scattering constants through code.

Support:

YAML and/or JSON configuration
CLI overrides where useful
Example conceptual configuration:

workload: type: agentic

precision: type: int4

speculation: enabled: true acceptance_rate: 0.8

controller: type: adaptive

memory: hbm: capacity_gb: 192 bandwidth_gbps: 3350 dram: capacity_gb: 1024 bandwidth_gbps: 1000 cxl: capacity_gb: 2048 bandwidth_gbps: 64 nvme: capacity_tb: 20 bandwidth_gbps: 10

These are examples only.

Choose physically/analytically coherent units and document them.

⸻

UNITS
Choose a consistent unit system.

Prefer explicit units such as:

bytes
GB
TB
bytes/sec
seconds
milliseconds
microseconds
tokens/sec
Do not mix arbitrary simulator units without documentation.

If historical results use arbitrary simulator units, label them as such.

⸻

HISTORICAL EXPERIMENTAL BASELINE
Earlier analytical simulation contained:

72 runs

6 scenarios × 4 controllers × 3 seeds.

Controllers:

Reactive
Static
Predictive
Oracle
Workloads included:

FP16
BitNet
INT4
with/without speculative decoding.

Earlier important historical observations:

FP16 speculative: Predictive P99: 55.092 → 45.344 ms

approximately -17.7%.

Throughput approximately +2.5%.

BitNet speculative: P99: 52.820 → 39.800 ms

approximately -24.6%.

Throughput: 6782 → 7663 tok/s

approximately +13%.

Oracle P99: 39.828 ms.

INT4 with 80% acceptance: P99: 34.630 → 26.174 ms

approximately -24.4%.

Throughput approximately +14.8%.

Negative cases:

FP16 autoregressive: approximately +9% P99.

BitNet autoregressive: approximately +16.2% P99.

INT4 with 20% acceptance: approximately +37.1% P99.

Historical predictive overhead:

average approximately 9.54–19.06 microseconds.

maximum P99 approximately 115.25 microseconds.

Historical simulated budget: <200 microseconds.

Historical BitNet speculative CXL stall: approximately 50.09%.

FP16 speculative CXL stall: approximately 19.70%.

These results suggested H6 inside that analytical simulation.

They do NOT constitute physical hardware validation.

⸻

IMPORTANT INTERPRETATION
The historical evidence supports a nuanced hypothesis:

Prediction is not universally beneficial.

Its value depends on:

workload
precision
speculative execution
memory pressure
interconnect pressure
prediction accuracy
confidence
prediction overhead
downstream synchronization
Therefore the final research goal is NOT:

“prove prediction wins.”

The goal is to determine:

“under what conditions prediction is beneficial, neutral, or harmful.”

⸻

RESEARCH OUTPUTS
The implementation should eventually produce:

Raw operation-level traces.
Run-level summaries.
Controller comparison tables.
Workload-specific reports.
Prediction quality reports.
Calibration reports.
Adaptive decision audit.
Oracle regret analysis.
Memory/interconnect utilization analysis.
Confidence-gated adaptive experiment.
H6 analysis.
H13 analysis.
H15 prediction evidence report.
Reproducibility report.
⸻

REPORT STRUCTURE
Reports should contain:

Executive Summary

What was tested.

Experimental Setup

Exact parameters.

Results

Tables and distributions.

Prediction Evidence

Accuracy/confidence/calibration.

Controller Behavior

Decision distribution.

Negative Results

Where prediction harmed performance.

Oracle Gap

Distance from theoretical upper bound.

Sensitivity Analysis

Which parameters matter.

Limitations

What the simulator does NOT prove.

Next Experiment

What should be tested next.

⸻

H6 ANALYSIS
The H6 experiment should vary arithmetic intensity / precision.

Compare at minimum:

FP16
INT4
BitNet
Measure:

compute fraction
memory fraction
transfer fraction
synchronization fraction
interconnect utilization
CXL pressure
performance sensitivity to memory placement
Do not simply assume the result.

Test it.

⸻

H13 ANALYSIS
The H13 experiment should compare:

Reactive
Predictive
WSA
Adaptive
Oracle
Across all workloads.

The key output is not merely “who is fastest.”

Analyze:

when Adaptive chooses prediction
when Adaptive remains reactive
why
confidence
prediction cost
downstream system cost
oracle regret
harmful decisions
⸻

H15 ANALYSIS
H15 should focus on whether the simulator provides meaningful evidence about prediction.

Do not call recorded placement indicators “top-k accuracy.”

Implement true prediction metrics where possible.

Include:

accuracy
confidence
calibration
uncertainty
entropy
NLL
Brier
ECE
prediction cost
where the underlying model supports them.

⸻

CODE QUALITY
Use:

type hints
docstrings
clear names
small functions
deterministic tests
explicit interfaces
structured logging
meaningful exceptions
Avoid:

giant monolithic files
duplicated constants
hidden global state
magic numbers
silent exception handling
⸻

DOCUMENTATION
Update README and documentation as implementation progresses.

Document:

architecture
equations
assumptions
parameters
workloads
controllers
metrics
experiments
limitations
The documentation must reflect the actual implementation.

⸻

AUTONOMOUS WORKFLOW
You are authorized to work autonomously through the implementation.

Follow this sequence:

PHASE 1: Inspect repository.

PHASE 2: Create architecture.

PHASE 3: Implement core models.

PHASE 4: Implement memory hierarchy.

PHASE 5: Implement workloads.

PHASE 6: Implement prediction.

PHASE 7: Implement controllers.

PHASE 8: Implement simulation engine.

PHASE 9: Implement metrics.

PHASE 10: Implement tests.

PHASE 11: Run smoke simulation.

PHASE 12: Run small deterministic benchmark.

PHASE 13: Audit results.

PHASE 14: Run larger experiments.

PHASE 15: Generate reports.

PHASE 16: Perform H6/H13/H15 analysis.

PHASE 17: Implement Confidence-Gated Adaptive.

PHASE 18: Run final controlled comparison.

⸻

IMPORTANT AGENT BEHAVIOR
Do not stop every few minutes asking for approval for ordinary implementation decisions.

Make reasonable engineering decisions yourself.

However, STOP and ask for clarification if:

a fundamental research assumption is ambiguous
two interpretations materially change the experiment
a required dependency cannot be installed
the environment cannot support the requested experiment
an experiment would require real hardware that is unavailable
a result appears inconsistent and needs human interpretation
Otherwise continue autonomously.

⸻

GIT WORKFLOW
Commit meaningful milestones.

Suggested commits:

Initial simulator architecture
Core simulation engine
Memory hierarchy
Workloads
Prediction subsystem
Controllers
Metrics and reporting
Tests and smoke benchmark
H6/H13 experiments
H15 prediction analysis
Confidence-Gated Adaptive
Final research report
Do not commit generated massive raw datasets unless necessary.

Use .gitignore appropriately.

⸻

DEPENDENCY MANAGEMENT
Prefer lightweight, stable Python dependencies.

Avoid unnecessary external services.

The simulator should run locally inside the Codespace.

Do not require paid APIs.

Do not require proprietary hardware.

If an optional dependency is useful, make it optional where practical.

⸻

FIRST EXECUTION
Start now.

First:

Inspect the repository.
Inspect README.
Inspect available Python/runtime environment.
Create the project architecture.
Implement the minimum viable core.
Add tests.
Run the smoke test.
Continue through the phases above.
At every stage:

show what changed
run relevant tests
report actual results
identify limitations
continue to the next stage
Do not invent results.

⸻

FINAL ACCEPTANCE CRITERIA
The project is considered successful only when:

simulator runs from a clean checkout
configuration works
all five controllers work
workloads work
memory hierarchy works
prediction works
confidence is explicit
speculative decoding works
metrics work
reproducibility works
tests pass
smoke benchmark passes
controlled benchmark runs
raw results are saved
reports are generated
negative results are retained
historical results are clearly separated from newly generated results
H6 is analyzed
H13 is analyzed
H15 is analyzed
Confidence-Gated Adaptive is implemented
limitations are documented
⸻

MOST IMPORTANT RULE
The objective is not to produce a convincing-looking result.

The objective is to build a scientifically useful simulator that can produce results that may SUPPORT, REFUTE, or QUALIFY the research hypotheses.

If the simulator shows that Adaptive performs worse than Reactive:

REPORT IT.

If prediction is harmful:

REPORT IT.

If H6 does not hold under a workload:

REPORT IT.

If the model is insufficient to answer a question:

REPORT IT.

Scientific validity is more important than reproducing historical results.

