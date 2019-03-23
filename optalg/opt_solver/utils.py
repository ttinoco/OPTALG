import os
import numpy as np

def cmd_exists(cmd):
    return any(os.access(os.path.join(path, cmd), os.X_OK)
               for path in os.environ["PATH"].split(os.pathsep))
