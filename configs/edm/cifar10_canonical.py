"""Canonical NVLabs EDM DDPM++/SongUNet recipe on CIFAR-10."""

from configs.vp.cifar10_ncsnpp_continuous import get_config as get_base_config
from configs.edm.default import apply_canonical_edm_defaults


def get_config():
  return apply_canonical_edm_defaults(get_base_config())
