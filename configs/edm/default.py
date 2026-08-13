"""Shared paper defaults for EDM baselines."""


def apply_edm_defaults(config):
  config.training.sde = 'edm'
  config.training.continuous = True
  config.training.reduce_mean = False
  config.training.likelihood_weighting = False
  config.training.edm_p_mean = -1.2
  config.training.edm_p_std = 1.2
  config.training.edm_augmentation = False

  config.sampling.method = 'edm'
  config.sampling.edm_num_steps = 18
  config.sampling.edm_sigma_min = 0.002
  config.sampling.edm_sigma_max = 80.0
  config.sampling.edm_rho = 7.0
  config.sampling.edm_s_churn = 0.0
  config.sampling.edm_s_min = 0.0
  config.sampling.edm_s_max = float('inf')
  config.sampling.edm_s_noise = 1.0

  config.data.centered = True

  config.model.name = 'edm_ncsnpp'
  config.model.scale_by_sigma = False
  config.model.embedding_type = 'fourier'
  config.model.edm_sigma_data = 0.5
  config.model.edm_ema_halflife_kimg = 500.0
  config.model.edm_ema_rampup_ratio = 0.05
  config.model.dropout = 0.13

  # Optimizer defaults from the official EDM training recipe.
  config.optim.lr = 1e-3
  config.optim.warmup = 0
  config.optim.warmup_kimg = 10000.0
  config.optim.grad_clip = -1.0
  config.optim.sanitize_gradients = True

  config.eval.enable_bpd = False
  config.eval.enable_sampling = True
  return config


def apply_canonical_edm_defaults(config):
  """Official NVLabs EDM DDPM++/SongUNet recipe at a 200M-image budget."""
  config = apply_edm_defaults(config)
  config.training.batch_size = 128
  config.training.effective_batch_size = 512
  config.training.gradient_accumulation_steps = 4
  # run_lib performs n_iters + 1 optimizer steps: 390625 * 512 = 200M images.
  config.training.n_iters = 390624
  config.training.snapshot_freq = 25000
  config.training.edm_augmentation = True
  config.training.edm_augment_probability = 0.12

  # Official EDM uses augmentation x-flips rather than dataset duplication.
  config.data.random_flip = False

  config.model.name = 'edm_canonical_songunet'
  config.model.embedding_type = 'positional'
  config.model.edm_official_root = 'third_party/edm'
  config.model.edm_official_commit = '008a4e5316c8e3bfe61a62f874bddba254295afb'
  config.model.edm_use_fp16 = False
  config.model.edm_model_channels = 128
  config.model.edm_channel_mult = (2, 2, 2)
  config.model.edm_num_blocks = 4
  config.model.edm_attn_resolutions = (16,)

  config.eval.begin_ckpt = 1
  config.eval.end_ckpt = 15
  config.eval.batch_size = 64
  return config
