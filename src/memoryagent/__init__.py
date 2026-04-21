"""memoryagent: joint REPLUG training of bge-small-en-v1.5 + Qwen3.5-0.8B."""

import os

# macOS: PyTorch and FAISS each link their own libomp, which segfaults at runtime
# unless we tell OpenMP to tolerate the duplicate. Must be set before either is
# imported, so it lives at the top of the package init.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
