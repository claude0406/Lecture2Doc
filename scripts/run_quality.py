"""Run archived baseline or revised algorithms; reuse immutable decode via hardlinks."""
import argparse
import importlib
from pathlib import Path
import shutil
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lecture2doc import pipeline
from lecture2doc.storage import digest, read, write
from lecture2doc.vision import Config
from accessmath_run import hardlink_or_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--decode-from',type=Path)
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--board',default='auto',choices=['auto','blackboard','whiteboard'])
    p.add_argument('--archive',type=Path,default=Path('experiments/accessmath-improvement/baseline-source'))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.decode_from:
        previous=read(args.decode_from/'manifest.json')
        if previous['source']['sha256']!=digest(args.video):
            raise ValueError('Decode source video hash mismatch')
        hardlink_or_copy(args.decode_from/previous['source']['path'],args.out/previous['source']['path'])
        shutil.copy2(args.decode_from/'source.json',args.out/'source.json')
        shutil.copytree(args.decode_from/'cache/decode',args.out/'cache/decode',dirs_exist_ok=True,copy_function=hardlink_or_copy)
        video=args.out/previous['source']['path']
    else:
        video=args.video
    if args.baseline:
        package=types.ModuleType('archived_quality_baseline')
        package.__path__=[str(args.archive.resolve())]
        sys.modules[package.__name__]=package
        old_vision=importlib.import_module(package.__name__+'.vision')
        old_reconstruct=importlib.import_module(package.__name__+'.reconstruct')
        pipeline.analyze=old_vision.analyze
        pipeline.scene_context=lambda *a,**k:{}
        pipeline.reconstruct=lambda root,folder,state,frames,config,context=None:old_reconstruct.reconstruct(root,folder,state,frames,config)
    code=(args.archive if args.baseline else Path('lecture2doc'))
    write(args.out/'algorithm.json',dict(variant='archived_baseline' if args.baseline else 'revised',
        sources={str(f):digest(f) for f in code.glob('*.py')},uses_ground_truth=False))
    result=pipeline.process(video,args.out,Config(board=args.board),progress=lambda s:print(s,flush=True))
    print('Complete',len(result['board_states']),'states',flush=True)


if __name__=='__main__':
    main()
