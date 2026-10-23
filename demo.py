"""Load Bazza and answer a new state/questions decision request."""
import argparse
import json
from pathlib import Path
import signal
import os
import sys
import torch
import time

from systemone import SystemOne
from utils import atomic_json, digest, file_hash, job_lock


def main(args):
    request = json.loads(args.request.read_text())
    if set(request) != {'state', 'questions'}:
        raise ValueError('Request requires exactly state and questions')
    backbone = args.backbone_path
    if backbone is None:
        if args.config is None:
            raise ValueError('Supply --backbone-path (or a Python --config with backbone_path)')
        from configuration import load_config
        backbone = load_config(args.config).backbone_path
    if args.plan_only:
        print(json.dumps({'checkpoint':str(args.checkpoint), 'backbone_path':str(backbone),
                          'request':request, 'device':args.device}, indent=2))
        return
    identity = {'checkpoint_sha256':file_hash(args.checkpoint), 'request_sha256':digest(request),
                'backbone_path':str(Path(backbone).resolve()), 'device':args.device,
                'max_length':args.max_length, 'torch':str(torch.__version__),
                'cuda':torch.version.cuda, 'image_id':os.environ.get('BAZZA_IMAGE_ID'),
                'backbone_files':{name:file_hash(Path(backbone)/name) for name in
                    ('config.json','tokenizer.json') if (Path(backbone)/name).exists()},
                'weight_file':({'size':(Path(backbone)/'model.safetensors').stat().st_size,
                               'mtime_ns':(Path(backbone)/'model.safetensors').stat().st_mtime_ns}
                               if (Path(backbone)/'model.safetensors').exists() else None),
                'sources':{str(p.relative_to(Path(__file__).parent)):file_hash(p) for p in
                    [Path(__file__), Path(__file__).with_name('systemone.py'),
                     Path(__file__).with_name('data.py'), Path(__file__).with_name('utils.py'),
                     *sorted(Path(__file__).parent.glob('model/*.py'))]}}
    def infer():
        print('Loading Bazza and the live backbone...', file=sys.stderr, flush=True)
        model = SystemOne.from_checkpoint(args.checkpoint, backbone_path=backbone,
                                         device=args.device, max_length=args.max_length)
        return model(**request)
    if args.output_dir is None:
        if args.resume:
            raise ValueError('--resume requires --output-dir')
        result = infer()
    else:
        root = args.output_dir
        root.mkdir(parents=True, exist_ok=True)
        with job_lock(root):
            plan = root/'request-plan.json'
            if plan.exists():
                if not args.resume or json.loads(plan.read_text()) != identity:
                    raise ValueError('Existing/different demo; use exact --resume or a fresh output directory')
            elif args.resume:
                raise ValueError('No request to resume')
            else:
                atomic_json(plan, identity)
                atomic_json(root/'request.json', request)
            if (root/'complete.json').exists():
                marker = json.loads((root/'complete.json').read_text())
                if marker['response_sha256'] != file_hash(root/'response.json'):
                    raise ValueError('Saved response checksum differs')
                result = json.loads((root/'response.json').read_text())
            else:
                atomic_json(root/'status.json', {'phase':'loading_model','updated_at':time.time()})
                def stop(signum, frame):
                    raise KeyboardInterrupt('signal '+str(signum))
                handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
                started=time.perf_counter()
                try:
                    result=infer()
                    atomic_json(root/'response.json', result)
                    atomic_json(root/'complete.json', {'response_sha256':file_hash(root/'response.json'),
                        'questions':len(result['answers']), 'seconds':time.perf_counter()-started})
                    atomic_json(root/'status.json', {'phase':'completed','updated_at':time.time()})
                except BaseException as error:
                    atomic_json(root/'status.json', {'phase':'stopped' if isinstance(error,KeyboardInterrupt) else 'failed',
                                                     'error':str(error),'updated_at':time.time()})
                    raise
                finally:
                    for sig, handler in handlers.items():signal.signal(sig,handler)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--backbone-path', type=Path)
    parser.add_argument('--request', type=Path, default=Path(__file__).parent/'examples/decision.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    main(args)
