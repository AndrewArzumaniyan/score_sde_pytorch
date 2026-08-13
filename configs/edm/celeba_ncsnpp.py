"""EDM-preconditioned NCSN++ baseline on 64x64 CelebA."""

from configs.vp.celeba_ncsnpp_continuous import get_config as get_base_config
from configs.edm.default import apply_edm_defaults


def get_config():
  return apply_edm_defaults(get_base_config())
