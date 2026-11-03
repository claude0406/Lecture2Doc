"""Temporary UI fixture: ticking status and a disposable trash record."""
from concurrent.futures import Future
from pathlib import Path
import time
import uvicorn
from lecture2doc.server import create_app
from lecture2doc.storage import write

root=Path('experiments/ui-editing/plain-preview')
running='e'*24
trash='f'*24
for identity,name,status in [(running,'시간 표시 확인용 작업','running'),(trash,'영구 삭제 확인용 임시 강의','complete')]:
 folder=root/identity;folder.mkdir(parents=True,exist_ok=True)
 write(folder/'source.json',dict(name=name))
 write(folder/'status.json',dict(id=identity,status=status,message='누적 판서 구간과 완성 원본 선정' if status=='running' else '완료',started_at=time.time()-75,updated_at=time.time(),elapsed_seconds=75))
 if identity==trash:
  (folder/'source.mp4').write_bytes(b'UI test fixture')
  write(folder/'manifest.json',dict(source=dict(name=name,path='source.mp4'),board_pages=[]))
write(root/'library-edits.json',dict(deleted={trash:dict(id=trash,name='영구 삭제 확인용 임시 강의',deleted_at='2026-11-03T00:00:00+00:00')}))
app=create_app(root)
app.state.futures[running]=Future()
app.state.run_clocks[running]=time.monotonic()-75
uvicorn.run(app,host='127.0.0.1',port=8891)
