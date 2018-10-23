import torch
import torch.nn as nn
import time
import seaborn
seaborn.set_context(context='talk')

from transformer import make_model

model = make_model(10, 10, 2)
print(model)


