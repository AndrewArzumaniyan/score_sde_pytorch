"""EDM-preconditioned NCSN++ controlled baseline on AFHQv2-64."""

from configs.edm.default import apply_edm_defaults
from configs.vp.afhqv2_ncsnpp_continuous import get_config as get_base_config


def get_config():
  config = apply_edm_defaults(get_base_config())
  # Karras et al. use 40 deterministic sampler steps for AFHQv2-64.
  config.sampling.edm_num_steps = 40
  return config
