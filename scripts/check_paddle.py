import importlib, sys
from importlib import util as _util

print('PYTHON:', sys.executable)
for pkg in ('paddle', 'paddleocr'):
    spec = _util.find_spec(pkg)
    print(pkg, 'FOUND' if spec else 'MISSING')
