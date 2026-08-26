"""Canonical NVLabs EDM DDPM++/SongUNet recipe on AFHQv2-64."""

from configs.edm.default import apply_canonical_edm_afhqv2_defaults
from configs.vp.afhqv2_ncsnpp_continuous import get_config as get_base_config


def get_config():
  return apply_canonical_edm_afhqv2_defaults(get_base_config())
