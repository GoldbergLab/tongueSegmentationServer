import glob
from pathlib import Path

if len(sys.argv) > 1:
    rootDir = sys.argv[1]
else:
    rootDir = '.'

completedDir = rootdir / 'Completed'
if not completedDir.exists():
    completedDir.mkdir()

p = Path(rootDir)
subdirs = [x for x in p.iterdir() if x.is_dir()]

for subdir in subdirs:
    
