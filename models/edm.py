"""EDM preconditioning wrapper around the repository's NCSN++ network."""

import torch
import torch.nn as nn

import edm_lib
from . import utils
from .ncsnpp import NCSNpp


@utils.register_model(name='edm_ncsnpp')
class EDMPreconditionedNCSNpp(nn.Module):
  """Apply the exact EDM denoiser preconditioning to NCSN++."""

  def __init__(self, config):
    super().__init__()
    if config.model.embedding_type.lower() != 'fourier':
      raise ValueError('EDM NCSN++ requires Fourier noise conditioning.')
    if config.model.scale_by_sigma:
      raise ValueError('EDM preconditioning requires model.scale_by_sigma=False.')
    self.sigma_data = float(config.model.edm_sigma_data)
    self.model = NCSNpp(config)

  def forward(self, x, sigma, augment_labels=None):
    if augment_labels is not None:
      raise ValueError('The controlled EDM-NCSN++ baseline does not use augmentation labels.')
    if not torch.is_tensor(sigma):
      sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
    sigma = sigma.to(device=x.device, dtype=x.dtype)
    if sigma.ndim == 0:
      sigma = sigma.expand(x.shape[0])
    sigma = sigma.reshape(-1)
    if sigma.shape[0] != x.shape[0]:
      raise ValueError('EDM sigma must be scalar or have one value per image.')
    sigma_view = sigma[:, None, None, None]
    c_skip, c_out, c_in, c_noise = edm_lib.EDM.preconditioning(
      sigma_view, self.sigma_data)

    # NCSN++'s Fourier path applies log() before its random projection.
    # Passing exp(c_noise) therefore supplies exactly EDM's c_noise=log(sigma)/4
    # to the Fourier features without modifying the shared architecture.
    network_output = self.model(c_in * x, torch.exp(c_noise.reshape(-1)))
    return c_skip * x + c_out * network_output
