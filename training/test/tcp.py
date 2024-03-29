import torch

print(torch.__version__)
import torch.distributed as dist
from datetime import timedelta

# store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
store = dist.TCPStore("localhost", 0, 1, True, timedelta(seconds=30))
store.add("first_key", 1)
