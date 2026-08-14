# Score-Based Generative Modeling through Stochastic Differential Equations

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/score-based-generative-modeling-through-1/image-generation-on-cifar-10)](https://paperswithcode.com/sota/image-generation-on-cifar-10?p=score-based-generative-modeling-through-1)

This repo contains a PyTorch implementation for the paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS)

by [Yang Song](https://yang-song.github.io), [Jascha Sohl-Dickstein](http://www.sohldickstein.com/), [Diederik P. Kingma](http://dpkingma.com/), [Abhishek Kumar](http://users.umiacs.umd.edu/~abhishek/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), and [Ben Poole](https://cs.stanford.edu/~poole/)

--------------------

We propose a unified framework that generalizes and improves previous work on score-based generative models through the lens of stochastic differential equations (SDEs). In particular, we can transform data to a simple noise distribution with a continuous-time stochastic process described by an SDE. This SDE can be reversed for sample generation if we know the score of the marginal distributions at each intermediate time step, which can be estimated with score matching. The basic idea is captured in the figure below:

![schematic](assets/schematic.jpg)

Our work enables a better understanding of existing approaches,  new sampling algorithms, exact likelihood computation, uniquely identifiable encoding, latent code manipulation, and brings new conditional generation abilities (including but not limited to class-conditional generation, inpainting and colorization) to the family of score-based generative models.

All combined, we achieved an FID of **2.20** and an Inception score of **9.89** for unconditional generation on CIFAR-10, as well as high-fidelity generation of **1024px** Celeba-HQ images (samples below). In addition, we obtained a likelihood value of **2.99** bits/dim on uniformly dequantized CIFAR-10 images.

![FFHQ samples](assets/ffhq_samples.jpg)

## Local extension

This repository was extended by **Arzumanyan Andrey** and **Kovshov Ilya** with a Fox-style colored-noise forward process for diffusion models.

In this fork we:

- added an exact Fox-reduced `FoxVPSDE` with constant linear drift and colored-noise kernels: Gaussian, Power-law, Mat\'ern $1/2$, and Mat\'ern $3/2$;
- integrated the new SDE into the continuous training and sampling pipeline with dedicated CIFAR-10 configs;
- added practical runtime compatibility fixes used in experiments, including fallbacks for custom CUDA ops and mixed TensorFlow/PyTorch startup issues.
- added controlled Cosine-VP baselines using the schedule from Improved DDPM;
- added an EDM-preconditioned NCSN++ baseline with the EDM loss, log-normal
  noise-level distribution, Karras grid, and Algorithm 2 Heun sampler.

### Cosine-VP and EDM baselines

The new baselines reuse the repository's training loop, logging, checkpoints,
snapshot samples, and IS/FID/KID evaluation artifacts.  Train them exactly like
the existing VP/VE/FOX configurations:

```sh
python main.py --mode=train \
  --config=configs/vp/cifar10_ncsnpp_cosine_continuous.py \
  --workdir=workdirs/cifar10_cosine_vp

python main.py --mode=train \
  --config=configs/edm/cifar10_ncsnpp.py \
  --workdir=workdirs/cifar10_edm
```

Two EDM variants are available:

- `configs/edm/*_ncsnpp.py` is the controlled EDM-preconditioned version of
  this repository's NCSN++;
- `configs/edm/*_canonical.py` uses the unmodified official NVLabs
  `EDMPrecond`, DDPM++-preset `SongUNet`, and `AugmentPipe`.

Install the separately licensed official implementation at the pinned commit
before using the canonical configs:

```sh
tools/install_official_edm.sh

python main.py --mode=train \
  --config=configs/edm/cifar10_canonical.py \
  --workdir=workdirs/cifar10_edm_canonical

python main.py --mode=train \
  --config=configs/edm/celeba_canonical.py \
  --workdir=workdirs/celeba_edm_canonical

python main.py --mode=eval \
  --config=configs/edm/cifar10_canonical.py \
  --workdir=workdirs/cifar10_edm_canonical --eval_folder=eval
```

Do this in the host checkout before starting a bind-mounted Docker run. The
repository mount hides the copy installed inside the image; installing from a
root container can also leave host files owned by root.

The canonical recipe uses an effective batch of 512 as four microbatches of
128, a 200M-image training budget, official 10M-image LR warmup, 500k-image
EMA half-life, and augmentation probability 0.12. The CelebA config transfers
the official EDM recipe to 64x64 CelebA; it is not a reproduction of a
published CelebA number from the EDM paper.

CelebA uses the matching configs
`configs/vp/celeba_ncsnpp_cosine_continuous.py` and
`configs/edm/celeba_ncsnpp.py`. Evaluation uses the same command and directory
layout as every other model:

```sh
python main.py --mode=eval --config=configs/edm/cifar10_ncsnpp.py \
  --workdir=workdirs/cifar10_edm --eval_folder=eval
```

Cosine-VP follows Nichol and Dhariwal's `s=0.008` cumulative alpha schedule.
The continuous SDE stops at `t=0.999` to avoid the mathematical singularity at
`t=1`; its remaining signal coefficient is approximately `1.56e-3`. EDM uses
the coefficients and defaults from Karras et al.: `sigma_data=0.5`,
`P_mean=-1.2`, `P_std=1.2`, `sigma_min=0.002`, `sigma_max=80`, and `rho=7`.
In the controlled EDM variant, the underlying image network remains this
repository's NCSN++, so it controls the image architecture, but its current
10M-image EDM warmup is not optimizer-matched to the older VP/FOX runs. The
canonical variant instead uses NVLabs' official SongUNet. EDM probability-flow
BPD is intentionally disabled; sample quality uses the common IS/FID/KID
protocol. The 18-step Heun sampler uses
`2 * steps - 1 = 35` network evaluations; when matching an NFE budget, set
`sampling.edm_num_steps` (or eval-only `eval.sampling_num_scales`) accordingly.
The official source is not vendored because its CC BY-NC-SA 4.0 license differs
from this repository's Apache license; see `THIRD_PARTY.md`.

### Reproducibility and artifact identity

`config.seed` is applied to Python, NumPy, PyTorch, TensorFlow, TFDS shuffle,
and stateless per-example preprocessing. Snapshot/eval RNG streams are isolated
from training; standalone sampling, loss, and BPD use `eval.sampling_seed`,
`eval.loss_seed`, and `eval.bpd_seed` respectively.

Each new training workdir contains `run_manifest.json`. Checkpoints are atomic,
periodic checkpoints are immutable, and incompatible code/config/seed resumes
fail instead of partially loading. A non-boundary final state is stored as
`checkpoint_final_step_<actual_step>.pth`; set
`eval.include_final_checkpoint=True` with the matching `training.n_iters` to
evaluate it. Each eval folder also has a manifest, so artifacts from another
checkpoint, sampler, seed, sample count, code version, or reference-statistics
file are not silently reused. Use a new eval folder after protocol changes.

The default contract fixes stochastic streams, data order, and artifact
provenance for an unchanged container, dataset, and GPU topology. It does not
guarantee identical weights or sample bytes, even on the same GPU, because the
canonical recipe keeps cuDNN benchmarking enabled. `deterministic=True` is a
strict fail-fast diagnostic mode and additionally requires `PYTHONHASHSEED`,
`TF_DETERMINISTIC_OPS=1` on the pinned TensorFlow, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` to be set before Python starts. Resume restores
the logical data position by deterministic replay/skip, which can be slow on a
long run. The official geometric augmentation can itself make strict mode stop
on a nondeterministic CUDA backward operation; disabling it would no longer be
the canonical recipe.

## What does this code do?
Aside from the **NCSN++** and **DDPM++** models in our paper, this codebase also re-implements many previous score-based models in one place, including **NCSN** from [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600), **NCSNv2** from [Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011), and **DDPM** from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). 

It supports training new models, evaluating the sample quality and likelihoods of existing models. We carefully designed the code to be modular and easily extensible to new SDEs, predictors, or correctors.

## **Integration with 🤗 Diffusers library**

Most models are now also available in 🧨 Diffusers and accesible via the [ScoreSdeVE pipeline](https://huggingface.co/docs/diffusers/api/pipelines/score_sde_ve).

Diffusers allows you to test score sde based models in PyTorch in just a couple lines of code.

You can install diffusers as follows:

```
pip install diffusers torch accelerate
```

And then try out the models with just a couple lines of code:

```python
from diffusers import DiffusionPipeline

model_id = "google/ncsnpp-ffhq-1024"

# load model and scheduler
sde_ve = DiffusionPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
image = sde_ve().images[0]


# save image
image[0].save("sde_ve_generated_image.png")
```

More models can be found directly [on the Hub](https://huggingface.co/models?library=diffusers&pipeline_tag=unconditional-image-generation&sort=downloads&search=ncsnpp).

## JAX version

Please find a JAX implementation [here](https://github.com/yang-song/score_sde), which additionally supports class-conditional generation with a pre-trained classifier, and resuming an evalution process after pre-emption.

###  JAX vs. PyTorch

In general, this PyTorch version consumes less memory but runs slower than JAX. Here is a benchmark on training an NCSN++ cont. model with VE SDE. Hardware is 4x Nvidia Tesla V100 GPUs (32GB)
| Framework | Time (second per step) | Memory usage in total (GB) |
|:----:|:----:|:----:|
|PyTorch | 0.56 | 20.6|
|JAX (`n_jitted_steps=1`)| 0.30 | 29.7 |
|JAX (`n_jitted_steps=5`) | 0.20 | 74.8|

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
pip install -r requirements.txt
```

### Stats files for quantitative evaluation

We provide the stats file for CIFAR-10. You can download [`cifar10_stats.npz`](https://drive.google.com/file/d/14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI/view?usp=sharing)  and save it to `assets/stats/`. Check out [#5](https://github.com/yang-song/score_sde/pull/5) on how to compute this stats file for new datasets.

### Usage

Train and evaluate our models through `main.py`.

```sh
main.py:
  --config: Training configuration.
    (default: 'None')
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <train|eval>: Running mode: train or eval
  --workdir: Working directory
```

* `config` is the path to the config file. Our prescribed config files are provided in `configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and should be quite self-explanatory.

  **Naming conventions of config files**: the path of a config file is a combination of the following dimensions:
  *  dataset: One of `cifar10`, `celeba`, `celebahq`, `celebahq_256`, `ffhq_256`, `celebahq`, `ffhq`.
  * model: One of `ncsn`, `ncsnv2`, `ncsnpp`, `ddpm`, `ddpmpp`.
  * continuous: train the model with continuously sampled time steps. 

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta checkpoints for pre-emption prevention, image samples, and numpy dumps of quantitative results.

* `mode` is either "train" or "eval". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist in `workdir/checkpoints-meta` . When set to "eval", it can do an arbitrary combination of the following

  * Evaluate the loss function on the test / validation dataset.

  * Generate a fixed number of samples and compute its Inception score, FID, or KID. Prior to evaluation, stats files must have already been downloaded/computed and stored in `assets/stats`.

  * Compute the log-likelihood on the training or test dataset.

  These functionalities can be configured through config files, or more conveniently, through the command-line support of the `ml_collections` package. For example, to generate samples and evaluate sample quality, supply the  `--config.eval.enable_sampling` flag; to compute log-likelihoods, supply the `--config.eval.enable_bpd` flag, and specify `--config.eval.dataset=train/test` to indicate whether to compute the likelihoods on the training or test dataset.

## How to extend the code
* **New SDEs**: inherent the `sde_lib.SDE` abstract class and implement all abstract methods. The `discretize()` method is optional and the default is Euler-Maruyama discretization. Existing sampling methods and likelihood computation will automatically work for this new SDE.
* **New predictors**: inherent the `sampling.Predictor` abstract class, implement the `update_fn` abstract method, and register its name with `@register_predictor`. The new predictor can be directly used in `sampling.get_pc_sampler` for Predictor-Corrector sampling, and all other controllable generation methods in `controllable_generation.py`.
* **New correctors**: inherent the `sampling.Corrector` abstract class, implement the `update_fn` abstract method, and register its name with `@register_corrector`. The new corrector can be directly used in `sampling.get_pc_sampler`, and all other controllable generation methods in `controllable_generation.py`.

## Pretrained checkpoints
All checkpoints are provided in this [Google drive](https://drive.google.com/drive/folders/1tFmF_uh57O6lx9ggtZT_5LdonVK2cV-e?usp=sharing).

**Instructions**: You may find two checkpoints for some models. The first checkpoint (with a smaller number) is the one that we reported FID scores in our paper's Table 3 (also corresponding to the FID and IS columns in the table below). The second checkpoint (with a larger number) is the one that we reported likelihood values and FIDs of black-box ODE samplers in our paper's Table 2 (also FID(ODE) and NNL (bits/dim) columns in the table below). The former corresponds to the smallest FID during the course of training (every 50k iterations). The later is the last checkpoint during training.

Per Google's policy, we cannot release our original CelebA and CelebA-HQ checkpoints. That said, I have re-trained models on FFHQ 1024px, FFHQ 256px and CelebA-HQ 256px with personal resources, and they achieved similar performance to our internal checkpoints. 

Here is a detailed list of checkpoints and their results reported in the paper. **FID (ODE)** corresponds to the sample quality of black-box ODE solver applied to the probability flow ODE.

| Checkpoint path | FID | IS | FID (ODE) | NNL (bits/dim) |
|:----------|:-------:|:----------:|:----------:|:----------:|
| [`ve/cifar10_ncsnpp/`](https://drive.google.com/drive/folders/1sP4GwvrYiI-sDPTp7sKYzsxJLGVamVMZ?usp=sharing) |  2.45 | 9.73 | - | - |
| [`ve/cifar10_ncsnpp_continuous/`](https://drive.google.com/drive/folders/1b0gy_LLgO_DaQBgoWXwlVnL_rcAUgREh?usp=sharing) | 2.38 | 9.83 | - | - |
| [`ve/cifar10_ncsnpp_deep_continuous/`](https://drive.google.com/drive/folders/11s6A_xM7qiztdj8AHQWqaIAUSC3I7uX2?usp=sharing) | **2.20** | **9.89** | - | - |
| [`vp/cifar10_ddpm/`](https://drive.google.com/drive/folders/1zDKcy3xbsN3F4AfyB_DfY_1oho89iKcf?usp=sharing) | 3.24 | - | 3.37 | 3.28 |
| [`vp/cifar10_ddpm_continuous`](https://drive.google.com/drive/folders/1RHNxW1qY-mTr0JMAE5t4V181Hi_aVWXK?usp=sharing) | - | - | 3.69| 3.21 |
| [`vp/cifar10_ddpmpp`](https://drive.google.com/drive/folders/1zOVj03ZBcq339p5QEKJPh2bBrxR_HOCM?usp=sharing) | 2.78 | 9.64 | - | - |
| [`vp/cifar10_ddpmpp_continuous`](https://drive.google.com/drive/folders/1xYjVMx10N9ivQQBIsEoXEeu9nvSGTBrC?usp=sharing) | 2.55 | 9.58 | 3.93 | 3.16 |
| [`vp/cifar10_ddpmpp_deep_continuous`](https://drive.google.com/drive/folders/1ZMLBiu9j7-rpdTQu8M2LlHAEQq4xRYrj?usp=sharing) | 2.41 | 9.68 | 3.08 | 3.13 |
| [`subvp/cifar10_ddpm_continuous`](https://drive.google.com/drive/folders/1DeebpmBkCxlZx89t3z45Te37T7BPOzd2?usp=sharing) | - | - | 3.56 | 3.05 |
| [`subvp/cifar10_ddpmpp_continuous`](https://drive.google.com/drive/folders/1bLgmnEAZnysRZfWt8qN3omGfijJ_B884?usp=sharing) | 2.61 | 9.56 | 3.16 | 3.02 |
| [`subvp/cifar10_ddpmpp_deep_continuous`](https://drive.google.com/drive/folders/16QGkviGcizSbIPRk37-YksUhlNIna4Ys?usp=sharing) | 2.41 | 9.57 | **2.92** | **2.99** |

| Checkpoint path | Samples |
|:-----|:------:|
| [`ve/bedroom_ncsnpp_continuous`](https://drive.google.com/drive/folders/18GmxDvfGR8se9uFucc9uweeVrX_GzuUG?usp=sharing) | ![bedroom_samples](assets/bedroom.jpeg) |
| [`ve/church_ncsnpp_continuous`](https://drive.google.com/drive/folders/1zVChA0HrnJU66Jkt4P6KOnlREhBMc4Yh?usp=sharing) | ![church_samples](assets/church.jpeg) |
| [`ve/ffhq_1024_ncsnpp_continuous`](https://drive.google.com/drive/folders/1ZqLNr_kH0o9DxvwSlrQPMmkrhEnXhBm2?usp=sharing) |![ffhq_1024](assets/ffhq_1024.jpeg)|
| [`ve/ffhq_256_ncsnpp_continuous`](https://drive.google.com/drive/folders/1KG72ZKUCUa8dDcA03hOf1BsnK8kBcdPD?usp=sharing) |![ffhq_256_samples](assets/ffhq_256.jpg)|
| [`ve/celebahq_256_ncsnpp_continuous`](https://drive.google.com/drive/folders/19VJ7UZTE-ytGX6z5rl-tumW9c0Ps3itk?usp=sharing) |![celebahq_256_samples](assets/celebahq_256.jpg)|


## Demonstrations and tutorials
| Link | Description|
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dRR_0gNRmfLtPavX2APzUggBuXyjWW55?usp=sharing)  | Load our pretrained checkpoints and play with sampling, likelihood computation, and controllable synthesis (JAX + FLAX)|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17lTrPLTt_0EDXa4hkbHmbAFQEkpRDZnh?usp=sharing) | Load our pretrained checkpoints and play with sampling, likelihood computation, and controllable synthesis (PyTorch) |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing) | Tutorial of score-based generative models in JAX + FLAX |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing)| Tutorial of score-based generative models in PyTorch |


## Tips
* When using the JAX codebase, you can jit multiple training steps together to improve training speed at the cost of more memory usage. This can be set via `config.training.n_jitted_steps`. For CIFAR-10, we recommend using `config.training.n_jitted_steps=5` when your GPU/TPU has sufficient memory; otherwise we recommend using `config.training.n_jitted_steps=1`. Our current implementation requires `config.training.log_freq` to be dividable by `n_jitted_steps` for logging and checkpointing to work normally.
* The `snr` (signal-to-noise ratio) parameter of `LangevinCorrector` somewhat behaves like a temperature parameter. Larger `snr` typically results in smoother samples, while smaller `snr` gives more diverse but lower quality samples. Typical values of `snr` is `0.05 - 0.2`, and it requires tuning to strike the sweet spot.
* For VE SDEs, we recommend choosing `config.model.sigma_max` to be the maximum pairwise distance between data samples in the training dataset.

## References

If you find the code useful for your research, please consider citing
```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```

This work is built upon some previous papers which might also interest you:

* Song, Yang, and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution." *Proceedings of the 33rd Annual Conference on Neural Information Processing Systems*. 2019.
* Song, Yang, and Stefano Ermon. "Improved techniques for training score-based generative models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.
* Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.
