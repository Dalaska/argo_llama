import argparse
# -----------------------------------------------------------------------------
# entrace to run the project

"""
These stages are designed to be run in order.
python run.py prep
python run.py train
python run.py viz
"""


parser = argparse.ArgumentParser()
parser.add_argument("stage", type=str, choices=[ "prep", "train", "viz"])
args = parser.parse_args()

# depending on the stage call the appropriate function

if args.stage == "prep":
    from prep import  prepare_data
    src_dir = '/home/dalaska/data/forecast_val/data'
    tar_dir = '/home/dalaska/val_new'
    prepare_data(src_dir, tar_dir)


elif args.stage == "train":
    from train import train
    DATA_DIR = "/home/dalaska/train_new"
    max_iters = 400000  # total number of training iterations
    train(DATA_DIR, max_iters)

elif args.stage == "viz":
    from viz import infer_and_viz
    DATA_DIR  = '/home/dalaska/val_new'
    ckpt_path = "ckpt.pt"
    infer_and_viz(DATA_DIR, ckpt_path)

else:
    raise ValueError(f"Unknown stage {args.stage}")
