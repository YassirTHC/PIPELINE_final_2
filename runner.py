import os
import sys
import subprocess
from pathlib import Path

# Répertoire du projet (celui du runner)
BASE_DIR = Path(__file__).parent.resolve()
VENV_PY = BASE_DIR / 'venv311' / 'Scripts' / 'python.exe'
MAIN_PY = BASE_DIR / 'main.py'

if __name__ == '__main__':
    # Transmettre tous les arguments à main.py
    args = sys.argv[1:]
    # Si on ne précise pas --cli mais qu'on donne une vidéo, on active --cli automatiquement
    if '--cli' not in args:
        args = ['--cli'] + args
    cmd = [str(VENV_PY), str(MAIN_PY)] + args
    # Assurer le cwd projet pour chemins relatifs (clips/, output/, temp/)
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    sys.exit(result.returncode) 