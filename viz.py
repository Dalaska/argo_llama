import glob
import pickle 
import os
import matplotlib.pyplot as plt
import numpy as np

color_map = { 1: 'r', 2: 'b', 3: [0.5 ,0.5, 0.5]}
def visualize(x, y, y_pred=None):
    # plot x
    for xx in  reversed(x):
        vect_type = xx[4]
        if vect_type == 0:
            continue
        color = color_map[vect_type]
        plt.arrow(xx[0], xx[1], xx[2]-xx[0], xx[3]-xx[1], head_width=2, head_length=2, fc=color, ec=color)
    # plot y
    color = 'm'
    for n in range(2):
        plt.arrow(y[2*n], y[2*n+1], y[2*n+2]-y[2*n], y[2*n+3]-y[2*n+1], head_width=3, head_length=3, fc=color, ec=color)
    # plot prediction
    if y_pred is not None:
        # print(y_pred)
        color = 'g'
        for n in range(2):
            plt.arrow(y_pred[2*n], y_pred[2*n+1], y_pred[2*n+2]-y_pred[2*n], y_pred[2*n+3]-y_pred[2*n+1], head_width=3, head_length=3, fc=color, ec=color)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.grid(True)  

def infer_and_viz(DATA_DIR, ckpt_path):
    print(f"DATA_DIR {DATA_DIR}")
    print(f"ckpt_path {ckpt_path}")

    if ckpt_path is not None:
        import torch
        from model import Transformer, ModelArgs
        from train import model_args
        device = "cpu"
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "input_width", "multiple_of", "input_length", "dropout"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    shard_filenames = glob.glob(os.path.join(DATA_DIR, "*.pkl"))


    for fn in shard_filenames:
        with open(fn, 'rb') as f:
            
            x, y = pickle.load(f)
            if ckpt_path is None:
                print(f"only viz {fn}")
                visualize(x, y)
                plt.savefig(f'{fn}.png') 
                plt.clf()
                plt.close()
                continue
            
            # do predict
            print(f"infer: {fn}")
            x_batch = torch.from_numpy(x)
            x_batch = x_batch.unsqueeze(0) 
            model.double()
            y_pred = model(x_batch, None)
            y_pred =y_pred.detach().numpy()
            y_pred = np.squeeze(y_pred)
            # print(y_pred)
            # print(y)

            visualize(x, y, y_pred)
            plt.savefig(f'{fn}.png') 
            plt.clf()
            plt.close()
    

if __name__ == '__main__':
    DATA_DIR = '/home/dalaska/train_pkl_test_hahaha'
    ckpt_path = None
    # ckpt_path = "ckpt.pt"
    infer_and_viz(DATA_DIR, ckpt_path)