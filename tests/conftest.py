import os
import sys

# Make `framework` importable from a plain checkout (no `pip install` needed).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
