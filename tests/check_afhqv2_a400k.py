"""Checks of the AFHQv2 A continuation to 400k: the launcher on stub docker, git and nvidia-smi,
and the facts of the training code it relies on. No GPU, no torch; runs on the host or in Docker."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
TRAIN_HEAD = '95de22d247b06b8b4c932a1de3a61a10046aa0a7'

DOCKER = r"""#!/usr/bin/env bash
last="${@: -1}"
case "$1" in
  image) echo sha256:image ;;
  ps) for f in "$STATE"/*.gpus; do [[ -e "$f" ]] && basename "$f" .gpus; done ;;
  inspect)
    [[ -e "$STATE/$last.gpus" ]] || { echo; exit 1; }
    case "$*" in
      *DeviceIDs*) cat "$STATE/$last.gpus" ;;
      *.Name*) echo "/$last" ;;
    esac ;;
  run)
    name=""; prev=""; for arg in "$@"; do [[ "$prev" == --name ]] && name="$arg"; prev="$arg"; done
    [[ -z "$name" ]] && name=check
    echo "run $name" >> "$STATE/events"; printf '%s\n' "$@" > "$STATE/$name.args" ;;
esac
exit 0
"""
STUBS = {
    'git': ('#!/usr/bin/env bash\nif [[ "$1" == -C ]]; then shift 2; [[ "$1" == rev-parse ]] && echo "${TRAIN_HEAD}"; '
            '[[ "$1" == status ]] && printf "%s" "${TRAIN_DIRTY:-}"; exit 0; fi\n[[ "$1" == rev-parse ]] && echo commit\n'
            'exit 0\n'),
    'nvidia-smi': ('#!/usr/bin/env bash\nif [[ "$*" == *name* ]]; then echo "${GPU_NAME:-NVIDIA RTX PRO 6000 Blackwell}"; '
                   'else echo "${FREE_MIB:-97000}"; fi\n'),
    'df': '#!/usr/bin/env bash\necho "Filesystem 1 2 3 4"\necho "d 0 0 999999999999999 0"\n',
    'realpath': '#!/usr/bin/env bash\npython3 -c "import os, sys; print(os.path.realpath(sys.argv[-1]))" "$@"\n',
}


class TrainingCode(unittest.TestCase):
    """The resume relies on these facts of afhqv2_rtx_train.py as committed in 95de22d."""

    def setUp(self):
        self.text = (REPO / 'afhqv2_rtx_train.py').read_text()

    def test_train_takes_any_target_on_the_50k_snapshots(self):
        self.assertIn('SNAPSHOT_FREQ = 50000', self.text)
        self.assertIn("a='workdirs/afhqv2_A_vp_pvp_rtx_seed42'", self.text)
        self.assertIn("elif len(sys.argv) == 4 and sys.argv[1] == 'train' and sys.argv[2] in ARMS:", self.text)
        self.assertIn("'--config.training.n_iters={}'.format(iters)", self.text)
        self.assertIn("return ROOT / WORK[arm] / 'checkpoints/checkpoint_{}.pth'.format(iters // SNAPSHOT_FREQ)", self.text)

    def test_n_iters_is_outside_the_training_protocol(self):
        text = (REPO / 'artifact_utils.py').read_text()
        excluded = re.search(r"for key in \('n_iters', 'log_freq', 'eval_freq',", text)
        self.assertIsNotNone(excluded)


class Launcher(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        self.bin, self.state, self.base = self.root / 'bin', self.root / 'state', self.root / 'base'
        for path in (self.bin, self.state, self.root / 'worktree'):
            path.mkdir()
        for name, text in dict(STUBS, docker=DOCKER).items():
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        shutil.copy(str(REPO / 'run_afhqv2_a400k_rtx.sh'), str(self.root / 'worktree'))
        main = self.base / 'score_sde_pytorch'
        self.source = self.base / 'afhqv2_rtx_20261108/workdirs/afhqv2_A_vp_pvp_rtx_seed42'
        for path in (main / 'assets/stats', main / 'datasets/afhqv2-64x64', self.source / 'checkpoints',
                     self.source / 'checkpoints-meta', self.base / 'score_sde_afhqv2_20261108'):
            path.mkdir(parents=True)
        for path in (main / 'assets/stats/afhqv2_64_cd017afab009_stats.npz', self.source / 'run_manifest.json',
                     self.source / 'checkpoints/checkpoint_4.pth', self.source / 'checkpoints-meta/checkpoint.pth'):
            path.write_text('source ' + path.name)
        self.output = self.base / 'afhqv2_a400k_20261113'

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           AFHQ_A400K_SERVER_BASE=str(self.base), **dict(dict(TRAIN_HEAD=TRAIN_HEAD), **env))
        return subprocess.run(['bash', str(self.root / 'worktree/run_afhqv2_a400k_rtx.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_start_copies_and_trains_from_the_old_worktree(self):
        result = self.launch('start', '2')
        self.assertEqual(result.returncode, 0, result.stdout)
        target = self.output / 'workdirs/afhqv2_A_vp_pvp_rtx_seed42'
        for path in ('run_manifest.json', 'checkpoints/checkpoint_4.pth', 'checkpoints-meta/checkpoint.pth'):
            self.assertEqual((target / path).read_text(), (self.source / path).read_text(), path)
        self.assertEqual(self.events(), ['run afhqv2-rtx-a400k'])
        args = (self.state / 'afhqv2-rtx-a400k.args').read_text()
        self.assertIn('score_sde_afhqv2_20261108,dst=/workspace/score_sde_pytorch,readonly', args)
        self.assertIn('afhqv2_a400k_20261113/workdirs,dst=/workspace/score_sde_pytorch/workdirs\n', args)
        self.assertIn('python -u afhqv2_rtx_train.py train a 400000', args)
        self.assertIn('/datasets/afhqv2-64x64,dst=/workspace/score_sde_pytorch/datasets/afhqv2-64x64,readonly', args)
        self.assertIn('device=2', args)
        self.assertIn('TORCH_CUDA_ARCH_LIST=12.0', args)
        self.assertNotIn('afhqv2_rtx_20261108/workdirs', args)  # The source run is never mounted.

    def test_rerun_keeps_what_the_training_wrote(self):
        self.launch('start', '2')
        meta = self.output / 'workdirs/afhqv2_A_vp_pvp_rtx_seed42/checkpoints-meta/checkpoint.pth'
        meta.write_text('state at 260k')
        (self.state / 'events').unlink()
        self.assertEqual(self.launch('start', '2').returncode, 0)
        self.assertEqual(meta.read_text(), 'state at 260k')

    def test_refuses_a_moved_or_changed_training_worktree(self):
        self.assertIn(b'is not at 95de22d', self.launch('start', '2', TRAIN_HEAD='59937bf').stdout)
        self.assertIn(b'has changed tracked files', self.launch('start', '2', TRAIN_DIRTY=' M run_lib.py').stdout)
        self.assertEqual(self.events(), [])

    def test_refuses_a_busy_or_wrong_card_and_bad_arguments(self):
        self.assertIn(b'not an RTX', self.launch('start', '3', GPU_NAME='NVIDIA H200 NVL').stdout)
        self.assertIn(b'need 60000', self.launch('start', '2', FREE_MIB='1000').stdout)
        (self.state / 'celeba-other.gpus').write_text('2,')
        self.assertIn(b'runs containers: celeba-other', self.launch('start', '2').stdout)
        (self.state / 'celeba-other.gpus').unlink()
        (self.state / 'a_ermilova_huggs.gpus').write_text('0,1,2,3,4,')
        self.assertEqual(self.launch('start', '2', '3').returncode, 2)
        self.assertEqual(self.launch('eval', '2').returncode, 2)
        self.assertEqual(self.events(), [])
        self.assertEqual(self.launch('start', '2').returncode, 0)  # Other people's containers do not count.

    def test_check_runs_these_tests_without_a_gpu(self):
        self.assertEqual(self.launch('check').returncode, 0)
        args = (self.state / 'check.args').read_text()
        self.assertIn('NVIDIA_VISIBLE_DEVICES=void', args)
        self.assertIn('tests/check_afhqv2_a400k.py', args)


if __name__ == '__main__':
    unittest.main()
