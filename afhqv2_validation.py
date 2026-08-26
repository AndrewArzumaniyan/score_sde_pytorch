"""Dependency-light validation of AFHQv2 configuration contracts."""

import unittest

from configs.edm.afhqv2_canonical import get_config as get_canonical_edm
from configs.edm.afhqv2_ncsnpp import get_config as get_controlled_edm
from configs.fox.afhqv2_ncsnpp_continuous import get_config as get_fox
from configs.fox.afhqv2_ncsnpp_continuous_matern_3_2 import get_config as get_matern
from configs.fox.afhqv2_ncsnpp_continuous_matern_3_2_vp_drift import get_config as get_vp_drift_matern
from configs.vp.afhqv2_ncsnpp_continuous import get_config as get_vp
from configs.vp.afhqv2_ncsnpp_cosine_continuous import get_config as get_cosine_vp


class AFHQv2ConfigValidationTest(unittest.TestCase):

  def test_all_variants_share_the_same_data_contract(self):
    configs = [
      get_vp(), get_cosine_vp(), get_controlled_edm(), get_canonical_edm(),
      get_fox(), get_matern(), get_vp_drift_matern(),
    ]
    for config in configs:
      self.assertEqual(config.data.dataset, 'AFHQV2')
      self.assertEqual(config.data.image_size, 64)
      self.assertEqual(config.data.num_channels, 3)
      self.assertEqual(config.data.afhqv2_train_take, -1)
      self.assertEqual(config.data.afhqv2_eval_take, -1)
      self.assertNotIn('celeba_dir', config.data)
      self.assertEqual(config.eval.num_samples, 50000)

  def test_method_specific_schedules_are_selected(self):
    self.assertEqual(get_vp().training.sde, 'vpsde')
    cosine = get_cosine_vp()
    self.assertEqual(cosine.training.sde, 'cosinevpsde')
    self.assertEqual(cosine.model.cosine_s, 0.008)
    self.assertEqual(get_fox().model.fox_kernel, 'gaussian')
    self.assertEqual(get_matern().model.fox_kernel, 'matern_3_2')
    vp_drift = get_vp_drift_matern()
    self.assertEqual(vp_drift.model.fox_drift_schedule, 'vp_linear')
    self.assertEqual(vp_drift.model.fox_beta_min, get_vp().model.beta_min)
    self.assertEqual(vp_drift.model.fox_beta_max, get_vp().model.beta_max)
    self.assertAlmostEqual(vp_drift.model.fox_matern_length_scale, 346.4101615138)
    self.assertEqual(get_controlled_edm().sampling.edm_num_steps, 40)

  def test_canonical_edm_matches_published_afhqv2_recipe(self):
    config = get_canonical_edm()
    self.assertEqual(config.training.sde, 'edm')
    self.assertEqual(config.model.name, 'edm_canonical_songunet')
    self.assertEqual(config.training.batch_size, 64)
    self.assertEqual(config.training.effective_batch_size, 256)
    self.assertEqual(config.training.gradient_accumulation_steps, 4)
    self.assertEqual((config.training.n_iters + 1) * 256, 200_000_000)
    self.assertEqual(tuple(config.model.edm_channel_mult), (1, 2, 2, 2))
    self.assertEqual(config.model.edm_model_channels, 128)
    self.assertEqual(config.model.edm_num_blocks, 4)
    self.assertEqual(tuple(config.model.edm_attn_resolutions), (16,))
    self.assertEqual(config.model.dropout, 0.25)
    self.assertEqual(config.training.edm_augment_probability, 0.15)
    self.assertEqual(config.optim.lr, 2e-4)
    self.assertEqual(config.model.edm_ema_halflife_kimg, 500.0)
    self.assertEqual(config.optim.warmup_kimg, 10000.0)
    self.assertEqual(config.sampling.edm_num_steps, 40)
    self.assertFalse(config.data.random_flip)


if __name__ == '__main__':
  unittest.main()
