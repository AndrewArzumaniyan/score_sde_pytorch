"""Published best-FID VP architecture and the matched CIFAR evaluation preset.

Training: use this file without a suffix (or with :train), in a new workdir.
The architecture/optimizer/loss/budget come from the author's deep continuous
VP config: eight blocks per resolution, about 950k updates, snapshots at 50k.
Snapshot sampling is disabled; training still samples uniform t, not uniform
log-SNR. The reported paper FID 2.41 used 50k generated images and a selected
checkpoint; it is not a guaranteed result of a new training run.

Evaluation: append :eval to the config filename. This selects the common
VP/FOX interval [-10, 9.115429865459795], 1000 EM steps + explicit Tweedie,
5000 images, batch 512, sampling seed 0. It defaults to checkpoint 8 (400k);
override eval.begin_ckpt/end_ckpt and sampling_seed on the command line.
The eval preset leaves every model field unchanged so the training checkpoint
can be restored. It is a screening protocol, not the paper's 50k evaluation.
"""

from configs.vp.cifar10_ddpmpp_deep_continuous import get_config as get_author_config


def get_config(name=None):
  if name not in (None, 'train', 'eval'):
    raise ValueError('Use this config without a suffix, with :train or with :eval.')
  config = get_author_config()

  # Explicitly preserve the historical VP/real-FOX training objective/clock.
  config.training.noise_distribution = 'uniform_time'
  config.model.noise_conditioning = 'time'
  config.training.snapshot_sampling = False

  if name == 'eval':
    config.sampling.time_grid = 'uniform_logsnr'
    config.sampling.logsnr_min = -10.0
    config.sampling.logsnr_max = 9.115429865459795
    config.sampling.noise_removal = True
    config.eval.endpoint_denoise_variants = True
    config.eval.sampling_num_scales = 1000
    config.eval.num_samples = 5000
    config.eval.batch_size = 512
    config.eval.sampling_seed = 0
    config.eval.enable_sampling = True
    config.eval.enable_loss = False
    config.eval.enable_bpd = False
    config.eval.begin_ckpt = 8
    config.eval.end_ckpt = 8

  return config
