import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest


def _install_tensorflow_stubs():
  if 'tensorflow' not in sys.modules:
    tf = types.ModuleType('tensorflow')
    tf.data = types.SimpleNamespace(
      experimental=types.SimpleNamespace(AUTOTUNE=object()))
    sys.modules['tensorflow'] = tf
  if 'tensorflow_datasets' not in sys.modules:
    tfds = types.ModuleType('tensorflow_datasets')
    tfds.core = types.SimpleNamespace(DatasetBuilder=object)
    tfds.ReadConfig = object
    sys.modules['tensorflow_datasets'] = tfds


class LocalCelebaLoaderTest(unittest.TestCase):

  def setUp(self):
    _install_tensorflow_stubs()
    self.datasets = importlib.import_module('datasets')
    self.tmpdir = tempfile.mkdtemp(prefix='celeba-local-')

  def tearDown(self):
    shutil.rmtree(self.tmpdir)
    os.environ.pop('CELEBA_DIR', None)

  def _write_minimal_celeba_tree(self, root):
    image_dir = os.path.join(root, 'img_align_celeba')
    os.makedirs(image_dir, exist_ok=True)
    for name in (
        'list_eval_partition.txt',
        'list_attr_celeba.txt',
        'list_landmarks_align_celeba.txt',
    ):
      with open(os.path.join(root, name), 'w') as fout:
        if name == 'list_eval_partition.txt':
          fout.write('000001.jpg 0\n')
          fout.write('000002.jpg 1\n')
          fout.write('000003.jpg 2\n')
        else:
          fout.write('header\n')

  def test_detects_direct_local_root(self):
    root = os.path.join(self.tmpdir, 'celeba')
    self._write_minimal_celeba_tree(root)

    normalized = self.datasets._normalize_local_celeba_root(root)

    self.assertEqual(normalized, root)
    self.assertEqual(self.datasets.get_local_celeba_split_paths('train', types.SimpleNamespace(
      data=types.SimpleNamespace(celeba_dir=root))), [os.path.join(root, 'img_align_celeba', '000001.jpg')])

  def test_detects_nested_local_root(self):
    outer = os.path.join(self.tmpdir, 'outer')
    nested = os.path.join(outer, 'celeba')
    self._write_minimal_celeba_tree(nested)

    normalized = self.datasets._normalize_local_celeba_root(outer)

    self.assertEqual(normalized, nested)

  def test_prefers_env_override(self):
    root = os.path.join(self.tmpdir, 'celeba')
    self._write_minimal_celeba_tree(root)
    os.environ['CELEBA_DIR'] = root

    resolved = self.datasets.get_local_celeba_root()

    self.assertEqual(resolved, root)


if __name__ == '__main__':
  unittest.main()
