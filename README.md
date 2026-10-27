<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Bayesian Kinematic Models of NFL Player Movement

> **Example project from Autumn 2025.** This is a copy of [MitchellVerhelle/bayes_final_project](https://github.com/MitchellVerhelle/bayes_final_project), shown with the authors' work, commit history and branches intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project).

Where will each player be a tenth of a second from now, and how sure can we be? The NFL Big Data Bowl 2026 on Kaggle asks for player positions after the ball is thrown, scored by root mean squared error. The team started from the physics of constant acceleration, put a probabilistic model around it, let the uncertainty depend on the player's position group, and drew the resulting predictive distributions on the field. A second model asks a different question of the same tracking data: will the targeted receiver catch the pass?

## Summary

A deterministic kinematic model predicts the next position from current position, speed, acceleration and direction. The Bayesian version keeps that prediction as the mean and learns the noise around it in PyMC, with half-normal priors on the x and y noise scales and NUTS sampling. A hierarchical version gives each position group, wide receivers, cornerbacks and so on, its own noise scale and systematic bias, drawn from global priors. Posterior predictive samples from these models are rolled forward and drawn as uncertainty cones and heatmaps on an animated field. Separately, a Bayesian logistic regression with receiver-specific random intercepts predicts whether a pass is completed from seven kinematic and spacing features, with prior and posterior predictive checks. On next-step prediction the Bayesian model matches the deterministic one, RMSE 0.54 against 0.537 yards, which is what the design implies: the two share a mean, and what the Bayesian model adds is a calibrated account of the spread.

## Team and contributions

Team "Adam's Family", Autumn 2025.

- **Mitch Verhelle** ([@MitchellVerhelle](https://github.com/MitchellVerhelle)): the package design, the kinematic and Bayesian kinematic models, the training pipeline, the field visualizations and animations, the play-success model class on `mitch_branch`, and most of the write-up.
- **Richard Huang** ([@rrhuang](https://github.com/rrhuang)): the hierarchical model and the heatmap visualization, the demo, and the write-up's hierarchical section.
- **Claire Fenton** ([@chelfen](https://github.com/chelfen)): the Bayesian logistic regression for play success in `bayesian_regression.ipynb`, with its predictive checks.

The project ran from December 2 to 10, 2025, on feature branches with a merged pull request. The attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/au25-nfl-player-movement/commits/main).

## Getting started

The environment is managed with uv and pinned in `uv.lock`; the authors used Python 3.13.

```sh
uv sync
```

Get the data as described under [Data](#data), then open the notebooks in this order:

| Notebook | What it does |
|---|---|
| `eda.ipynb` | Explores plays with animations, then fits the deterministic baseline and the Bayesian kinematic model on weeks 1 to 3 and draws predictions with 95 percent cones. |
| `eda_2.ipynb` | Fits the hierarchical model on week 1 and renders the heatmap animation that `demo.html` and `demo_frames/` were made from. |
| `bayesian_regression.ipynb` | Builds play-level features, fits the Bayesian logistic regression, and runs prior and posterior predictive checks. |

The models are importable from the `models` package and follow one interface: `fit`, `predict_dataframe`, `rmse`, and for Bayesian models `posterior_samples_for_rows`. `tests/test_abstracts.py` exercises that interface with dummy models, and `pipeline.py` builds step-level datasets from the weekly files and trains and evaluates a model on chosen weeks.

## Data

Data files are never committed to this repository. Git ignores every CSV.

- **Source:** the [NFL Big Data Bowl 2026 – Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) competition on Kaggle: 2023 season tracking data at ten frames per second, with positions, speed, acceleration, orientation and direction for every player, split into the frames before the throw (input) and after it (output), for 18 weeks.
- **Terms:** Kaggle competition rules. The data is for competition and non-commercial use and must not be redistributed, which is why it isn't here.
- **How to get it:** join the competition on Kaggle, accept the rules, and download the files.
- **Where to put it:** the weekly files `input_2023_w01.csv` to `w18` and `output_2023_w01.csv` to `w18` in `train/`, and `test_input.csv` and `test.csv` in the repository root.

The `kaggle_evaluation/` folder is Kaggle's own submission harness for the competition, included as provided.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: data dictionary, the three models with equations and code, visualizations, RMSE. |
| `models/model_base.py`, `bayes_model_base.py` | Abstract base classes: a common prediction interface, RMSE evaluation, multi-step rollouts, posterior sampling. |
| `models/kinematic.py`, `kinematic_boundary.py` | The deterministic baseline, and a variant that keeps players on the field. |
| `models/bayes_kinematic.py` | The Bayesian kinematic model. |
| `models/bayes_kinematic_hierarchical.py` | The hierarchical version with position and optional player levels. |
| `models/bayes_kinematic_hmm.py` | A regime-mixture model; see the note under [what to learn](#what-to-learn-from-this-project). |
| `models/pipeline.py` | Step-level dataset construction and train-and-evaluate helpers over weeks. |
| `tools/` | Field drawing, play animation, prediction cones and heatmaps, animation export. |
| `tests/` | Tests of the model interface. |
| `demo.html`, `demo_frames/`, `output2_cropped.mp4` | The hierarchical model's heatmap animation. |
| Branch `mitch_branch` | `models/play_success_bayes.py`, the play-success model as a class, and its notebook. |

## Method

**Kinematics.** With speed $s$, acceleration $a$, direction $\theta$ and time step $\Delta t = 0.1$ s, the next position is the current one plus $\bar v \cos\theta\,\Delta t$ and $\bar v \sin\theta\,\Delta t$, where $\bar v$ is the average of the current speed and the speed after accelerating for one step.

**Bayesian kinematic model.** The observed next position is Normal around that prediction, with separate noise scales for x and y. Both scales have HalfNormal(1) priors, in yards. NUTS, two chains of 1,000 draws after 1,000 tuning steps, or ADVI when speed matters. For prediction, the posterior means of the two scales are plugged in and future positions are sampled around the kinematic mean, then rolled forward several frames.

**Hierarchical model.** Each position group gets its own noise scales, Normal around the global scales with spread 0.3, and its own bias in x and y, Normal(0, 0.2). A player level nested inside positions is implemented and off by default. The heatmaps come from kernel density estimates of posterior samples at one and two frames ahead, clipped to the field.

**Play success.** A play counts as successful when the targeted receiver ends up close to where the ball lands, about 18 percent of plays; the write-up gives the threshold as half a yard in one place and a meter in another. Seven standardized features, the receiver's final speed, acceleration and direction, path length, path curvature, and the mean and final distance to the nearest defender, enter a logistic regression with a receiver-specific random intercept whose scale has an Exponential(1) hyperprior. NUTS with four chains; prior and posterior predictive checks with ArviZ.

### What is Bayesian here

The kinematic equations are a physical model with no free parameters; the Bayesian layer treats what physics leaves out as noise with a distribution and learns that distribution from data. The half-normal priors say the noise is positive and probably under a yard, and the posterior over the noise scales is what a single number could not give: a statement of how predictable a player's next step is. The hierarchical model is partial pooling in its textbook form. Position groups with little data are pulled toward the global scale, groups with a lot of data move away from it, and the biases let the model learn, for example, that a position group systematically outruns or undershoots its kinematic prediction. The logistic regression's random intercepts do the same for receivers. The prior and posterior predictive checks in the regression notebook are the course's model-criticism material applied properly: simulate from the prior to see whether it is sensible, simulate from the posterior to see whether the model reproduces the data.

## Results

Reported by the authors in the write-up.

| Model | Next-step RMSE (yards) |
|---|---|
| Deterministic kinematic | 0.537 |
| Bayesian kinematic | 0.54 |

The near-equality is the honest outcome: both models predict the same mean. The Bayesian model's contribution is the predictive distribution, shown in the write-up as posterior predictive densities that lie on top of the observed next positions, and in the animations as cones and heatmaps that widen as predictions are rolled further ahead. No RMSE is reported for the hierarchical model. For play success, the write-up shows the posterior coefficient distributions and how the predicted probability varies with the nearest-defender distance, the receiver's distance to the landing spot, and the defenders' tackle range.

## What to learn from this project

Course staff added this section. The engineering and the Bayesian reasoning are both sound; the points below are about evidence and naming.

- **Put the numbers where they can be checked.** The two RMSE values were added to the write-up by hand in the final commit. No committed notebook computes them, and the write-up doesn't say which weeks were held out. The pipeline's default evaluates on the last training week, which would be a leak. A results notebook with the split stated would have settled this.
- **Report every model you compare.** The hierarchical model is the project's most interesting idea and has no number next to it. Did position-specific noise and bias reduce RMSE, or only change the spread? The repository can't say.
- **Name models by what they do.** The file called an HMM defines a transition matrix and never uses it; the regime weights are a softmax over speed and acceleration, so it is a feature-gated mixture, a reasonable model under the wrong name. The write-up doesn't advertise it, so the fix is in the code.
- **Keep scales positive.** The hierarchical model's position-level noise scales have Normal priors and are passed through an absolute value. A HalfNormal or LogNormal prior on the deviation would say what is meant.
- **Use the posterior, not just its mean.** Prediction plugs in the posterior means of the noise scales. Sampling the scales from the posterior along with the positions would propagate the uncertainty the model worked to estimate. It would change little here, because the posteriors are tight, but it is the right habit.
- **Small things.** The play-success model class lives on an unmerged branch; `pyproject.toml` is a generic environment named "clinic-env" with about 150 unpinned dependencies, several unrelated to the project, pinned only through `uv.lock`; the `LICENSE` file is the MIT template with its placeholders and web page front matter still in place, so it doesn't yet grant anything.

What the project does well: a model interface that made four models comparable with one evaluator; a deterministic baseline reported alongside the Bayesian models rather than beaten by them; priors with units and reasons; hierarchical structure used for a real grouping; predictive checks before and after fitting; and uncertainty made visible on the field, which is the whole point of the Bayesian layer.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Autumn 2025. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository, with all branches, and made these edits. The authors' code and notebooks are unchanged.

- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.
- Removed two `.DS_Store` files and added them to `.gitignore`.

## License

The repository's `LICENSE` file is the MIT template with the copyright holder left blank. Treat the code as the authors' and ask them before reusing it; they can complete the license with their names.
