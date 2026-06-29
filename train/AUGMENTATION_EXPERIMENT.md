# ACT Data-Augmentation Experiment

## Course-Requirement Check

This experiment addresses the following requirements:

- Establish an unaugmented baseline and attempt an improvement.
- Apply appropriate image transforms and analyze their impact.
- Use proprioceptive normalization. ACT already applies `MEAN_STD`
  normalization to state and action features through the saved policy
  preprocessor and postprocessor.
- Report quantitative success rates and qualitative success/failure videos.

It does **not** replace the requirement to design at least 2-3 manipulation
tasks. At present, `task1_grab` is one task. A minimum practical completion plan
is to add two simpler variants, each with 50+ demonstrations:

- Task 2: pick the cube and place it in a marked target area.
- Task 3: sort two visually distinct objects into two target areas.

Those tasks can use their own baseline policies; the detailed augmentation
ablation below can remain focused on `task1_grab`.

## Motivation

The course requires an unaugmented baseline, an attempted improvement, and an
analysis of how image transforms affect policy generalization.

The existing `act_task1_grab` model is the baseline:

- ACT with ResNet18
- 60 episodes / 35,989 frames
- 50,000 steps, batch size 4, seed 1000
- deterministic resize to 360x640 only
- no random data augmentation

The experiment changes only the image transforms. Dataset, policy, optimizer,
seed, batch size, and training steps remain fixed.

## Training Conditions

| ID | Training transforms | Purpose |
|---|---|---|
| E0 Baseline | Resize(360x640) | Existing reference model |
| E1 Photo | Resize + mild ColorJitter | Improve lighting/color robustness |
| E2 Photo+Geo | Resize + mild ColorJitter + mild RandomAffine | Add robustness to small camera/object shifts |

The augmentation strengths are deliberately mild because manipulation requires
precise spatial information. Horizontal flips are not used because they would
change the relationship between visual positions and robot actions.

In LeRobot 0.4.4, `max_num_transforms` randomly selects transforms. The provided
script sets it equal to the number of configured transforms so Resize is always
applied; otherwise the two camera streams may retain different dimensions.

## AutoDL Training

First run 100-step smoke tests:

```bash
cd /root/autodl-tmp/ECE371-final-lerobot-so101

AUG=photo STEPS=100 SAVE_FREQ=100 JOB=act_aug_photo_smoke \
  bash train/train_act_aug.sh

AUG=photo_geo STEPS=100 SAVE_FREQ=100 JOB=act_aug_photo_geo_smoke \
  bash train/train_act_aug.sh
```

Then train both experiments:

```bash
AUG=photo bash train/train_act_aug.sh \
  2>&1 | tee /root/autodl-tmp/act_aug_photo.log

AUG=photo_geo bash train/train_act_aug.sh \
  2>&1 | tee /root/autodl-tmp/act_aug_photo_geo.log
```

Expected output directories:

```text
/root/autodl-tmp/outputs/train/act_task1_grab_aug_photo
/root/autodl-tmp/outputs/train/act_task1_grab_aug_photo_geo
```

## Real-Robot Evaluation

Evaluate E0, E1, and E2 using the same robot calibration, initial arm pose,
camera placement, episode duration, and task success definition.

Run 5 trials per model under each condition:

| Condition | Description |
|---|---|
| C0 In-distribution | Original lighting, background, and cube region |
| C1 Lighting shift | Clearly brighter or darker lighting |
| C2 Position shift | Cube starts several centimeters outside the usual region |
| C3 Background shift | Add a harmless visual distractor or change the table background |

This produces `3 models x 4 conditions x 5 trials = 60 trials`.

For every trial, record:

```text
model, condition, trial, success, completion_time_s, failure_type, notes
```

Use a binary success definition decided before testing, for example:

> Success means the gripper lifts the cube clear of the table and holds it for
> at least two seconds within the episode time limit.

Recommended failure labels:

- `missed_cube`
- `failed_grasp`
- `dropped_cube`
- `unsafe_motion`
- `timeout`

## Metrics

Primary metric:

```text
Success rate = successful trials / total trials
```

Also report:

- Mean completion time among successful trials
- Success rate for each condition
- Generalization gap: `C0 success rate - mean(C1, C2, C3 success rate)`
- Failure-type counts
- Representative success and failure videos

Training loss alone must not be used to claim improved generalization. An
augmented model may have higher training loss but better real-world success.

## Interpretation

- E1 better than E0 under C1: photometric augmentation helped lighting robustness.
- E2 better than E1 under C2/C3: mild geometric augmentation added useful robustness.
- E2 worse than E1 under C0: geometric augmentation may have reduced precision.
- No meaningful improvement: augmentation strength, data coverage, or task diversity
  may be limiting factors; this is still a valid experimental result if analyzed.
