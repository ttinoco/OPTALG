import os
import sys
import numpy as np

def cmd_exists(cmd):
    if 'win32' in sys.platform.lower():
        cmd += '.exe'
    return any(os.access(os.path.join(path, cmd), os.X_OK)
               for path in os.environ["PATH"].split(os.pathsep))
