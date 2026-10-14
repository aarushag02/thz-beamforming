import pathlib
import sys

# Make `import src...` work from pytest, notebooks and plain scripts alike.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
