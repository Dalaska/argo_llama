"""
modify model args here!!!!
"""
# model
dim = 96 #5 #in vector space
n_layers = 6
n_heads = 12 #5
n_kv_heads = 12 #5
multiple_of = 32
dropout = 0.0

# data
batch_size = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
input_length = 244
input_width = 5  # input 3->30 # the Llama 2 tokenizer has 32K tokens #output size


# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    input_width=input_width,
    multiple_of=multiple_of,
    input_length=input_length,
    dropout=dropout,
    )  # start with model_args from command line
