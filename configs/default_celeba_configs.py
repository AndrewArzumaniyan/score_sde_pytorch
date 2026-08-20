import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 1300001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.17
  sampling.time_grid = 'uniform_time'

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 26
  evaluate.batch_size = 1024
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.sampling_num_scales = 0
  evaluate.sampling_seed = 0
  evaluate.loss_seed = 0
  evaluate.bpd_seed = 0
  evaluate.include_final_checkpoint = False

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CELEBA'
  data.celeba_dir = 'celeba'
  # Deterministic prefix-of-the-split subset for smoke/overfit probes; -1 (the
  # default) means the full split. CelebA has no class labels to balance a
  # subset by, unlike the CIFAR-10 per-class knobs, so this just takes the
  # first N paths from list_eval_partition.txt for that split.
  data.celeba_train_take = -1
  data.celeba_validation_take = -1
  data.image_size = 64
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 90.
  model.sigma_min = 0.01
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.deterministic = False
  config.cudnn_benchmark = False
  config.allow_tf32 = False
  config.allow_fp16_reduced_precision_reduction = False
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config
