# Why a better detector built a worse map

All numbers below are produced by `erl-map sweep --scene conf/scenes/room_v1.yaml`
and re-derived in CI on every push. Nothing here is quoted from a run that no
longer exists.

## The observation

| detector | frame accuracy | map mIoU | mIoU (observed) | ECE | mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| `independent:0.7` | 69.9% | 0.560 | 0.769 | 0.028 | 0.92 |
| `independent:0.85` | 84.9% | 0.617 | 0.903 | 0.023 | — |
| `viewbias:0.85` | 85.0% | 0.502 | 0.674 | 0.100 | 0.96 |

The detector with the *higher* per-frame accuracy produced the *lower* map
quality. Ranking two perception front-ends by their per-frame benchmark score
would have chosen the worse one.

## The mechanism

Per-voxel semantic fusion accumulates log-likelihoods:

```
log P(c | z₁..z_n)  ∝  log P(c) + Σᵢ log P(zᵢ | c)
```

That factorisation is the whole method, and it holds only if the observations
`z₁..z_n` are **conditionally independent given the voxel's class `c`**. When
they are, `n` observations carry `n` observations' worth of evidence and the
posterior concentrates on the truth. When they are not, the sum still grows
linearly in `n` — it just grows toward whatever the errors agree on.

The two detectors are constructed to differ in exactly this and nothing else:

- **`independent:p`** — each observation is correct with probability `p`,
  drawn independently. The assumption is satisfied exactly.
- **`viewbias:p`** — error is a deterministic function of
  `(true class, viewing sector)` over 16 sectors (8 azimuth × 2 elevation). A
  surface observed twice from the same sector is mislabelled *identically* both
  times. The assumption is violated maximally: conditioned on the class and the
  sector, the observations have zero variance.

Because voxels are surfaces, they are visible from a narrow range of directions.
Each occupied voxel in `room_v1` receives about 7 observations, and most of them
arrive from one or two sectors. Under `viewbias`, those observations are not 7
independent votes; they are one vote counted 7 times. Fusion reads the
repetition as corroboration.

This is why the failure is *confident* rather than merely wrong. The posterior's
concentration is driven by `n`, and `n` is real — the rays genuinely arrived. It
is the information content that was overcounted. `viewbias:0.85` reports a mean
confidence of **0.96** with an ECE of **0.100**, against **0.023** for
`independent:0.85` at the same accuracy.

## Ruling out the alternative explanation

The obvious objection is that `viewbias` is simply a harder noise model — that
15% of its errors hurt more than 15% of anyone else's. The matched-accuracy pair
answers it:

| | `independent:0.85` | `viewbias:0.85` |
| --- | ---: | ---: |
| realised frame accuracy | 84.9% | 85.0% |
| map mIoU | 0.617 | 0.502 |
| ECE | 0.023 | 0.100 |

Same scene, same trajectory, same rays, same number of observations (77,864),
and the same number of *wrong* observations to within a tenth of a percentage
point. The only difference is whether the wrong ones agree with each other. That
is worth 0.115 mIoU.

The matching is not incidental. `viewbias` is calibrated against the actual
observation histogram of the run: corrupting a `(class, sector)` pair that
carries 30% of all rays is a completely different detector from one that carries
0.3%, so the pairs to corrupt are selected to hit the requested error rate on
this run's distribution. Without that step the comparison would be confounded by
error *rate* rather than isolating error *structure*.

Geometry is paired the same way. Ray casting depends only on the seed, so it
runs once and every detector in the sweep consumes the identical set of hits.
The test suite asserts that all eight sweep configurations recorded the same
observation count.

## Where the damage lands

| detector | floor | wall | table | chair | shelf | box |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `independent:0.85` | 0.85 | 0.97 | 0.48 | 0.45 | 0.52 | 0.44 |
| `viewbias:0.85` | 0.79 | 0.83 | 0.27 | 0.24 | 0.57 | 0.31 |
| delta | −0.06 | −0.14 | **−0.21** | **−0.21** | +0.05 | **−0.13** |

The distribution of harm follows the geometry directly. Floors and walls are
large surfaces observed from many sectors, so even correlated error gets partial
averaging and degrades gently. Free-standing objects are small and observed from
a narrow wedge of directions — the regime where the independence assumption
fails hardest — and they lose roughly a fifth of their IoU.

`shelf` *improves* by 0.05, and the reason is worth stating rather than hiding:
with 16 sectors and 6 classes, which pairs get corrupted is a discrete draw, and
`shelf` happened to have few of its high-traffic pairs selected at this seed. The
per-class effect is noisy; the aggregate is not, which is why the CI gate is
written against mIoU and ECE rather than against any single class.

This matters practically because the objects a semantic map exists to support —
grasping, placing, obstacle-specific planning — are exactly the small ones.
**A correlated detector degrades most on the classes the map is built for**, and
degrades least on the classes that dominate the aggregate metric. A single mIoU
number partially masks its own worst case.

## What to take from this

1. **Per-frame segmentation accuracy does not order map quality.** Two detectors
   with the same accuracy differed by 0.115 mIoU here; a detector 15 points
   *worse* on accuracy beat one that was better.
2. **Report calibration next to accuracy.** ECE separated the two matched
   detectors by 4x. It is the signal that survives when the accuracy number does
   not, and it is what a downstream planner needs in order to distrust a voxel.
3. **The independence assumption is testable, and mostly untested.** For a real
   segmentation model, bin per-frame errors by viewing angle relative to the
   surface and check whether the conditional error rates are flat. If they are
   not, log-odds fusion is overcounting evidence by roughly the ratio of
   observations to *effective* independent observations.
4. **A fix exists and is cheap.** Down-weight observations by an effective sample
   size per sector rather than counting each ray as independent evidence. This
   repo does not implement it — the point here is that you cannot know you need
   it from a per-frame benchmark.

## Limits

The scenes are synthetic, and that is load-bearing rather than incidental: the
claim is about correlation structure, and testing it needs exact ground truth
plus the ability to hold accuracy fixed while varying only that structure.
Neither is available from a real annotated dataset.

What this does **not** establish is the magnitude of the effect for any
particular real detector. `viewbias` is an extreme: its errors are perfectly
correlated within a sector. Real segmentation error is somewhere between the two
models here, and where it sits is an empirical question about a specific network
on a specific dataset. The contribution is that the question is well posed, has a
measurable answer, and is invisible to the benchmark most people run.
