import torch
from model import Transformer, ModelArgs
from model_args_file import batch_size, model_args, input_length, input_width

gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
rand_input = torch.rand(batch_size, input_length, input_width)
output_dim = 6
targets = torch.rand(batch_size, output_dim)
output = model(rand_input, targets)
print(output)
print(output.shape)
