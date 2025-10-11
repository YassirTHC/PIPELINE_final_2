import sys
if getattr(sys.stdout, "closed", False):
    sys.stdout = getattr(sys, "__stdout__", sys.stdout)
if getattr(sys.stderr, "closed", False):
    sys.stderr = getattr(sys, "__stderr__", sys.stderr)
