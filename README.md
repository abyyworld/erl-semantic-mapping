# erl-semantic-mapping

**Semantic occupancy mapping with an evaluation harness that gates on its own assumptions.**

Bayesian label fusion over a posed depth sensor, plus a benchmark that refuses to
report a number it cannot defend: every published metric is re-derived in CI, and
the build fails if the suite stops discriminating or the headline finding stops
reproducing.

```bash
git clone https://github.com/abyyworld/erl-semantic-mapping
cd erl-semantic-mapping
docker build -t erl-semantic-mapping .
docker run --rm erl-semantic-mapping     # the whole gate, ~3 s, no GPU, no data download
```

Or without Docker:

```bash
make install
make check      # re-derive the suite's assumptions and the finding
make sweep      # the full detector x fusion matrix
```

Downstream of [`erl-vla-evals`](https://github.com/abyyworld/erl-vla-evals) and
[`erl-teleop-pipeline`](https://github.com/abyyworld/erl-teleop-pipeline), which
apply the same standard to policy evaluation and to training data.

---

## The finding this repo exists for

**A semantic detector with 85% per-frame accuracy builds a worse map than one
with 70% per-frame accuracy**, on the same scene, the same trajectory, and the
same rays.

| detector | frame accuracy | map mIoU | ECE |
| --- | ---: | ---: | ---: |
| `independent:0.7` | 69.9% | **0.560** | 0.028 |
| `viewbias:0.85` | 85.0% | **0.502** | 0.100 |

The cause is not accuracy, it is the *shape* of the error. Bayesian fusion
multiplies likelihoods, which is only valid if observations are conditionally
independent given the voxel's class. `independent` satisfies that and its errors
average away over the ~7 observations each voxel receives. `viewbias` does not:
its errors are a deterministic function of `(true class, viewing sector)`, so a
surface seen repeatedly from one direction is mislabelled *identically* every
time. Fusion reads the repetition as corroboration and converges — confidently —
on the wrong class.

Controlling for accuracy isolates the mechanism:

| detector | frame accuracy | map mIoU | ECE |
| --- | ---: | ---: | ---: |
| `independent:0.85` | 84.9% | 0.617 | 0.023 |
| `viewbias:0.85` | 85.0% | 0.502 | 0.100 |

Same accuracy, **0.115 mIoU apart**, and calibration error **4x worse**. The
biased map is not just wrong, it reports a mean confidence of 0.96 while being
wrong — which is the failure mode that a downstream planner cannot detect.

Per-class IoU shows exactly where it lands:

| detector | floor | wall | table | chair | shelf | box |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `independent:0.85` | 0.85 | 0.97 | 0.48 | 0.45 | 0.52 | 0.44 |
| `viewbias:0.85` | 0.79 | 0.83 | **0.27** | **0.24** | 0.57 | **0.31** |

Large surfaces survive: floors and walls are seen from many sectors, so even
correlated error gets some averaging. Free-standing objects are the casualty —
they are visible from a narrow range of directions, which is precisely the
regime where the independence assumption fails hardest. That is the practical
consequence: **the correlated detector degrades most on exactly the small,
graspable, manipulation-relevant objects a semantic map is built for.**

Full derivation: [`docs/CORRELATED_ERROR.md`](docs/CORRELATED_ERROR.md).

## Why this is not an artefact of the benchmark

A result like the above is only interesting if the benchmark could have shown
otherwise. Four properties are checked on every push, and each one fails the
build:

| check | what it rules out |
| --- | --- |
| a perfect detector scores **1.000** mIoU on observed voxels | the mapper or metric is lossy on its own, capping every result below |
| a uniform-random detector scores **0.066** mIoU (chance is 0.167) | labels can be stumbled into, making any score partly free |
| fusing 77,864 observations beats using the first by **+0.253** mIoU | fusion is decorative and the comparison measures nothing |
| the trajectory observes **87.2%** of occupied voxels | the map is being graded on a corner of the room |

Three more re-derive the finding itself, so the README cannot drift away from
what the code does:

```
[PASS] accuracy does not order map quality: viewbias at 85.0% frame accuracy maps to
       0.502 mIoU, worse than independent at 69.9% which maps to 0.560
[PASS] correlation is the cause, not accuracy: at matched 84.9% accuracy, independent
       errors give 0.617 mIoU and correlated errors 0.502 (+0.114, need >= 0.050)
[PASS] correlated error is confidently wrong: ECE is 0.100 under correlated error vs
       0.023 under independent error at the same accuracy (need >= 2x)
```

`make check` runs all seven and exits non-zero on any failure.

## What this does

### 1. Semantic occupancy mapping

Log-odds occupancy with free-space carving, and per-voxel semantic fusion over a
symmetric confusion model. Five fusion strategies, so the contribution of each
piece is measurable rather than asserted:

| detector | fusion | frame acc | mIoU | mIoU (obs) | occ IoU | coverage | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | bayes | 100.0% | 0.656 | 1.000 | 0.872 | 87.2% | 0.019 |
| random | bayes | 16.5% | 0.066 | 0.070 | 0.872 | 87.2% | 0.495 |
| independent:0.7 | bayes | 69.9% | 0.560 | 0.769 | 0.872 | 87.2% | 0.028 |
| independent:0.85 | bayes | 84.9% | 0.617 | 0.903 | 0.872 | 87.2% | 0.023 |
| viewbias:0.85 | bayes | 85.0% | 0.502 | 0.674 | 0.872 | 87.2% | 0.100 |
| independent:0.7 | first | 69.9% | 0.306 | 0.348 | 0.872 | 87.2% | 0.000 |
| independent:0.7 | majority | 69.9% | 0.561 | 0.772 | 0.872 | 87.2% | 0.257 |
| independent:0.7 | none | 69.9% | 0.000 | 0.000 | 0.872 | 87.2% | 0.000 |

Two rows in that table are worth dwelling on.

**`majority` scores 0.561 against `bayes`'s 0.560 — and that is not noise, it is
an identity.** Under a symmetric confusion matrix, each observation adds the same
constant to every class's log-likelihood plus a fixed bonus to the observed one,
so the log-likelihood argmax *is* the vote count argmax. Bayes fusion buys no
accuracy here; what it buys is the posterior. Majority voting's confidence (vote
fraction) has an ECE of **0.257** against Bayes's **0.028**. A test pins the
equivalence so that the day someone introduces a class-dependent confusion model,
it fails loudly instead of silently becoming false.

**`none` scores 0.000 mIoU at 87.2% coverage.** A mapper that discards semantics
does not get to report low coverage and look merely incomplete; it reports full
coverage and zero quality, which is what it earned.

### 2. Metrics that do not flatter the map

- **`semantic_miou`** is computed over *every* ground-truth occupied voxel, with
  unobserved voxels counted as errors. **`semantic_miou_observed`** restricts to
  what was seen. The gap between them (0.656 vs 1.000 for a perfect detector) is
  the credit a mapper would otherwise get for the scene it never looked at —
  here, mostly the interiors of solid objects, which no surface sensor can reach.
  The oracle row is therefore the ceiling: **read every score relative to 0.656,
  not to 1.0.**
- **Absent classes are skipped, not scored as zero.** Averaging a missing class
  in as 0.0 silently penalises a correct map.
- **ECE** is reported next to mIoU throughout, because the interesting failure in
  this repo is not being wrong, it is being wrong and confident.
- **Occupancy IoU** is reported separately from semantics, and free-space carving
  is tested against ground truth, so a mapper cannot inflate it by marking
  everything occupied.

### 3. Runs that are actually comparable

The scene and the trajectory are pure functions of the seed, and ray casting
happens once per run and is shared by every detector. Two runs that differ only
in detector therefore see **identical geometry and identical viewing
directions** — the sweep asserts that all eight configurations recorded the same
77,864 observations. Without that pairing, a 0.115 mIoU difference could just be
two different scenes.

`RunConfig` also rejects a sensor placed outside the room. That configuration
still produces a full set of rays and a plausible-looking map, so it is invisible
in the metrics; it just quietly makes every number meaningless.

## Reproducibility

Three things, in increasing order of how often they are skipped:

1. **Pinned dependencies.** `requirements.lock` and `requirements-dev.lock` are
   hash-locked and installed with `--require-hashes --no-deps`, so a re-uploaded
   wheel fails the build rather than silently changing a result. CI re-runs
   `pip-compile` and fails if the locks have drifted from `pyproject.toml`.
2. **A digest-pinned base image.** `python:3.12-slim@sha256:57cd7c3a…`, not a tag
   that moves under you. The runtime stage carries no compiler, no pip and no
   build context, and runs as a non-root user.
3. **CI that runs the eval, not just the tests.** Every push rebuilds the sweep
   from scratch, posts the metrics table to the job summary, gates against a
   committed baseline at a 0.02 mIoU tolerance, and separately builds the
   container and runs the same gate inside it — because "one command runs it
   anywhere" is a claim, and claims get tested.

## Commands

```bash
erl-map run viewbias:0.85 --scene conf/scenes/room_v1.yaml --tag biased
erl-map sweep --scene conf/scenes/room_v1.yaml    # full matrix -> results/
erl-map check --scene conf/scenes/room_v1.yaml    # all 7 gates, non-zero on failure
erl-map compare baselines/room_v1.json results/candidate.json --tolerance 0.02
erl-map scenarios                                 # what the sweep runs
```

Detectors: `oracle`, `random`, `independent:<p>`, `viewbias:<p>`.
Fusion: `bayes`, `majority`, `first`, `last`, `none`.

`viewbias` is calibrated against the observation distribution of the run it is
used in, so its realised accuracy matches the requested one. Without that step
the comparison would be confounded — corrupting a `(class, sector)` pair carrying
30% of all rays is a very different detector from one carrying 0.3%.

Under Docker, mount a volume to keep the output:

```bash
docker run --rm -v "$PWD/results:/work/results" erl-semantic-mapping sweep --scene conf/scenes/room_v1.yaml
```

## Scope

The scenes are synthetic. That is a deliberate trade, and the reason is in the
finding: the claim is about the *correlation structure* of detector error, and
establishing it needs exact ground truth plus the ability to hold accuracy fixed
while varying only that structure. No real dataset offers either — annotations
are noisy, and you cannot order a segmentation network to make 15% errors that
are correlated by viewing angle. What synthetic data cannot tell you is the
magnitude of the effect on any particular real detector; it tells you the effect
exists, why, and what to measure to detect it.

The natural next step is to measure sector-conditioned error rates on a real
segmentation model over a posed dataset, and check whether the independence
assumption its fusion relies on is one it actually satisfies.

## Licence

MIT.
