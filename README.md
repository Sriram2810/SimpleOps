Meet SimpleOps

A static MLOperations enabled purely with DVC only.

Note:

-> Install uv
-> uv sync
-> Activate .venv

Always activate .venv before performing any activity.

Please run the command below to utilize the GPU compatibility of the torch.
uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

Utilizing uv add torch will lead to installing the cpu version only.
