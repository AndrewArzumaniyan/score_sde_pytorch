# coding=utf-8
"""k_a(t) shape experiment on CelebA, all arms trained and evaluated on one GPU type.

k_a(t) = k_bar + a (t - 1/2), k_bar = -5.025, Matern 3/2 kappa=1000, q(1) = 1.
int_0^1 k_a = k_bar for every a, so lambda(1) = -10.05 is shared and only the
path between the ends changes.  Select the arm after the colon:

  --config=configs/fox/celeba_kaffine.py:a0          constant drift, control
  --config=configs/fox/celeba_kaffine.py:ap5         a = +5, front-loaded, raw scale
  --config=configs/fox/celeba_kaffine.py:vplin_norm  a = -9.95 (vp_linear path), r == 1
  --config=configs/fox/celeba_kaffine.py:vplin_raw   a = -9.95, raw scale (r up to 2.13)
  --config=configs/fox/celeba_kaffine.py:ap9p95      a = +9.95, mirror of vp_linear

Training and evaluation use the same config; sampling fields do not enter the
loss.  Evaluation: uniform log-SNR grid on the common window [-10, 10.5757],
EM + explicit Tweedie (endpoint_denoise_variants).
"""

from configs.fox.celeba_ncsnpp_continuous_affine import get_config as get_affine_config

# arm -> (fox_drift_a, fox_normalize_scale)
ARMS = {
  'a0': (0.0, False),
  'ap5': (5.0, False),
  'vplin_norm': (-9.95, True),
  'vplin_raw': (-9.95, False),
  'ap9p95': (9.95, False),
}
COMMON_LOGSNR_MIN = -10.0
COMMON_LOGSNR_MAX = 10.575749165617202
N_ITERS = 100000
SNAPSHOT_FREQ = 25000


def get_config(arm='a0'):
  if arm not in ARMS:
    raise ValueError(f'Unknown k_a arm {arm!r}; expected one of {sorted(ARMS)}.')
  drift_a, normalize_scale = ARMS[arm]
  config = get_affine_config()

  model = config.model
  model.fox_drift_a = drift_a
  model.fox_normalize_scale = normalize_scale

  training = config.training
  training.batch_size = 128
  training.n_iters = N_ITERS
  training.snapshot_freq = SNAPSHOT_FREQ
  training.snapshot_freq_for_preemption = 5000
  training.snapshot_sampling = False
  training.log_freq = 50
  training.eval_freq = 500
  training.kaffine_arm = arm

  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'
  sampling.time_grid = 'uniform_logsnr'
  sampling.logsnr_min = COMMON_LOGSNR_MIN
  sampling.logsnr_max = COMMON_LOGSNR_MAX
  sampling.noise_removal = True

  evaluate = config.eval
  evaluate.endpoint_denoise_variants = True
  evaluate.include_final_checkpoint = False
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.enable_sampling = True
  evaluate.batch_size = 512
  evaluate.sampling_num_scales = 1000

  config.seed = 42
  config.allow_tf32 = True
  config.cudnn_benchmark = True
  return config
