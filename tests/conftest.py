import os

# Mirrors src/memoryagent/__init__.py — pytest discovers tests before importing
# memoryagent, so we need the workaround set here too.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
