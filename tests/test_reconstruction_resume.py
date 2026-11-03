"""A missing aggregate manifest must not discard completed board checkpoints."""
from pathlib import Path

import cv2
import numpy as np

from lecture2doc import pipeline


def test_resume_uses_completed_boards(tmp_path,monkeypatch):
    video=tmp_path/'video.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),4,(320,180))
    for _ in range(12):
        writer.write(np.full((180,320,3),180,np.uint8))
    writer.release()
    def analyze(root,frames,config,duration):
        half=len(frames)//2
        states=[]
        for i,group in enumerate([frames[:half],frames[half:]]):
            states.append(dict(id=f'b{i:05d}',scene=i,start=group[0]['time'],
                end=frames[half]['time'] if i==0 else duration,
                source_frame_ids=[f['id'] for f in group],selected_frame_id=group[-1]['id'],
                selection_reason='test_input',boundary_reason='test_input',quality_flags=[],transcript_ids=[]))
        return dict(states=states,metrics=[])
    calls=[]
    def restore(root,folder,state,frames,config,context=None):
        calls.append(state['id'])
        return dict(id='r'+state['id'][1:],board_state_id=state['id'],status='fallback',
                    image=frames[state['selected_frame_id']]['image'],source_frame_ids=[state['selected_frame_id']],
                    quality_flags=[],coverage=None,provenance=None)
    monkeypatch.setattr(pipeline,'analyze',analyze)
    monkeypatch.setattr(pipeline,'reconstruct',restore)
    out=tmp_path/'record'
    first=pipeline.process(video,out)
    assert len(calls)==2
    folder=next((out/'cache/reconstruction').iterdir())
    # Simulate interruption after the first board completed but before stage completion.
    (folder/'stage.json').unlink()
    (folder/'b00001.result.json').unlink()
    calls.clear()
    resumed=pipeline.process(video,out)
    assert calls==['b00001']
    assert resumed['reconstructed_boards']==first['reconstructed_boards']
