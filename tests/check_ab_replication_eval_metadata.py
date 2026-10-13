import ast, hashlib, json, os, subprocess, sys, types
from pathlib import Path
repo=Path('/source/score_sde_pytorch_ours')
root=Path('/workspace/score_sde_pytorch'); root.mkdir(parents=True)
training=Path('/workspace/training_workdirs'); training.mkdir()
stubs=Path('/tmp/eval_stubs'); stubs.mkdir()
(root/'logs').mkdir(); (root/'assets/stats').mkdir(parents=True)
(root/'main.py').write_text('# fixture source\n')
reference=root/'assets/stats/celeba_64_0c2d89311192_stats.npz'
reference.write_bytes(b'reference fixture')
environment={key:'fixture-version' for key in ('python','numpy','tensorflow','torch','torch_cuda','cudnn')}
environment['packages']={}
(stubs/'ml_collections.py').write_text('''class ConfigDict(dict):
 def __init__(self, value=None):
  super().__init__({k:ConfigDict(v) if isinstance(v,dict) else v for k,v in (value or {}).items()})
 def __getattr__(self,k): return self[k]
 def __setattr__(self,k,v): self[k]=v
 def to_dict(self): return {k:v.to_dict() if isinstance(v,ConfigDict) else v for k,v in self.items()}
''')
(stubs/'torch.py').write_text('''import json
class CUDA:
 def is_available(self): return True
 def device_count(self): return 1
 def init(self): pass
cuda=CUDA()
def device(value): return value
def load(path, **kwargs): return json.loads(path.read_text())
''')
(stubs/'evaluation.py').write_text('def _get_stats_filename(config): return "assets/stats/celeba_64_0c2d89311192_stats.npz"\n')
artifact_source=(repo/'artifact_utils.py').read_text()
artifact=ast.parse(artifact_source)
names={'_canonicalize','_source_fingerprint','_manifest_with_hash','build_model_manifest','build_training_manifest'}
funcs=[n for n in artifact.body if isinstance(n,ast.FunctionDef) and n.name in names]
(stubs/'artifact_utils.py').write_text('import hashlib, json, math, os\nTRAINING_MANIFEST_SCHEMA_VERSION=1\nclass NP:\n generic=type("Generic",(),{})\nnp=NP()\n'+'\n\n'.join(ast.get_source_segment(artifact_source,fn) for fn in funcs)+'\n'+f'def _environment_manifest(): return {environment!r}\n')
sys.path.insert(0,str(repo))
sys.path.insert(0,str(stubs))
from artifact_utils import _source_fingerprint, build_model_manifest, build_training_manifest
from ml_collections import ConfigDict
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.factorial.celeba_b_vp_pfox import get_config as get_b
manifests={}; states={}
for arm,name in [('A','A_vp_pvp'),('B','B_vp_pfox')]:
 path=training/f'celeba_ab_replication_{name}_seed43_200k'; (path/'checkpoints').mkdir(parents=True)
 config=(get_a if arm=='A' else get_b)()
 config.seed=43; config.training.n_iters=200000; config.training.batch_size=128
 config.training.snapshot_freq=50000; config.training.snapshot_sampling=False
 manifest=build_training_manifest(config, repo_root=str(root))
 manifest['source_sha256']='ddf4c6acbc51b67da7a3465f2b90678ff559a628f6925d47e991350b02ff54f7'
 assert 'n_iters' not in manifest['config']['training'] and 'eval' not in manifest['config']
 state={'step':200001,'training_protocol_sha256':manifest['training_protocol_sha256'],
 'model_protocol_sha256':build_model_manifest(ConfigDict(manifest['config']))['model_protocol_sha256'],
 'checkpoint_id':f'fixture-checkpoint-{arm}'}
 manifests[arm]=(path/'run_manifest.json',manifest); states[arm]=(path/'checkpoints/checkpoint_4.pth',state)
 (path/'run_manifest.json').write_text(json.dumps(manifest)); (path/'checkpoints/checkpoint_4.pth').write_text(json.dumps(state))
env={**os.environ,'PYTHONPATH':str(stubs)+os.pathsep+str(repo),'CUDA_VISIBLE_DEVICES':'0',
 'AB_REPLICATION_EVAL_SOURCE_SHA256':_source_fingerprint(str(root)),
 'AB_REPLICATION_EVAL_REFERENCE_SHA256':hashlib.sha256(reference.read_bytes()).hexdigest()}
text=(repo/'run_ab_replication_eval.sh').read_text()
blocks=[block.split('\nPY\n',1)[0] for block in text.split("<<'PY'\n")[1:]]
def preflight(): return subprocess.run([sys.executable,'-c',blocks[0]],cwd=root,env=env,capture_output=True,text=True)
good=preflight(); assert good.returncode==0, good.stderr
assert json.loads((root/'logs/ab_replication_eval_seed0_preflight.json').read_text())['status']=='passed'
print('PASS: eval metadata preflight accepts matched manifests/checkpoints at step200001')
for key,value,message in [('step',200000,'identity/step'),('training_protocol_sha256','wrong','identity/step'),('model_protocol_sha256','wrong','identity/step')]:
 path,state=states['B']; broken={**state,key:value}; path.write_text(json.dumps(broken))
 result=preflight(); assert result.returncode!=0 and message in result.stderr, result.stderr
 path.write_text(json.dumps(state))
print('PASS: incomplete/swapped checkpoint identities stop evaluation')
path,manifest=manifests['B']; broken=json.loads(json.dumps(manifest)); broken['config']['seed']=42; path.write_text(json.dumps(broken))
result=preflight(); assert result.returncode!=0 and 'Unexpected training manifest' in result.stderr
path.write_text(json.dumps(manifest))
(root/'main.py').write_text('# changed source\n')
result=preflight(); assert result.returncode!=0 and 'sources changed' in result.stderr
(root/'main.py').write_text('# fixture source\n')
reference.write_bytes(b'changed reference')
result=preflight(); assert result.returncode!=0 and 'reference changed' in result.stderr
reference.write_bytes(b'reference fixture')
print('PASS: wrong training seed and source/reference changes stop evaluation')
factory={}; exec(blocks[1],factory)
for arm in ('A','B'):
 config=factory['get_config'](arm)
 assert config.seed==43 and config.training.n_iters==200000
 assert config.sampling.logsnr_min==-10.0 and config.sampling.logsnr_max==9.115429865459795
 assert config.eval.endpoint_denoise_variants and not config.eval.include_final_checkpoint
print('PASS: ephemeral eval configs preserve real seed43 training configs and set common endpoint')
mockbin=Path('/tmp/eval_bin'); mockbin.mkdir()
spy=Path('/tmp/spy_eval_main.py'); spy.write_text('''import json,runpy,sys
from pathlib import Path
args=sys.argv[1:]
config_arg=next(x.split("=",1)[1] for x in args if x.startswith("--config="))
path,arm=config_arg.rsplit(":",1)
config=runpy.run_path(path)["get_config"](arm)
assert config.seed==43
for flag in ("--mode=eval","--config.eval.begin_ckpt=4","--config.eval.end_ckpt=4","--config.eval.batch_size=512","--config.eval.num_samples=10000","--config.eval.sampling_num_scales=1000","--config.eval.sampling_seed=0","--config.eval.enable_loss=False","--config.eval.enable_bpd=False"):
 assert flag in args,flag
with Path("/tmp/eval_calls.jsonl").open("a") as out: out.write(json.dumps({"arm":arm,"args":args})+"\\n")
''')
(mockbin/'python').write_text('''#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == '-u -' ]]; then exec /usr/bin/python "$@"; fi
[[ "$1" == -u && "$2" == main.py ]]
shift 2
exec /usr/bin/python /tmp/spy_eval_main.py "$@"
'''); (mockbin/'python').chmod(0o755)
env.update(PATH=f'{mockbin}:'+os.environ['PATH'],TFDS_DATA_DIR='/tmp/eval_tfds',TORCH_EXTENSIONS_DIR='/tmp/eval_extensions',XDG_CACHE_HOME='/tmp/eval_cache')
run=subprocess.run(['bash',str(repo/'run_ab_replication_eval.sh')],cwd=root,env=env,capture_output=True,text=True)
assert run.returncode==0, run.stderr
calls=[json.loads(line) for line in Path('/tmp/eval_calls.jsonl').read_text().splitlines()]
assert [x['arm'] for x in calls]==['A','B']
print('PASS: eval shell routes A then B to exact common10k/ckpt4/seed0 protocol')
print('Metadata tests use stub Torch/CUDA and JSON fixture checkpoints; no sampling or metrics are executed.')
