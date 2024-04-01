import argparse
import os
# -----------------------------------------------------------------------------
# entrace to run the project

"""
These stages are designed to be run in order.
python run.py prep
python run.py train
python run.py viz
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=[ "prep", "train", "viz"])
    args = parser.parse_args()
    current_directory = os.getcwd()
    # # depending on the stage call the appropriate function
    if args.stage == "prep":
        from prep import  prepare_data
        src_dir = os.path.join(current_directory, 'sample/data/csv')
        tar_dir = os.path.join(current_directory, 'sample/data/pkl')
        prepare_data(src_dir, tar_dir)


    elif args.stage == "train":
        from train import train
        DATA_DIR = os.path.join(current_directory, 'sample/data/pkl')
        max_iters = 100000  # total number of training iterations
        train(DATA_DIR, max_iters)

    elif args.stage == "viz":
        from viz import infer_and_viz
        DATA_DIR  = os.path.join(current_directory, 'sample/data/pkl')
        ckpt_path = os.path.join(current_directory, 'sample/ckpt_260k.pt')
        infer_and_viz(DATA_DIR, ckpt_path)

    else:
        raise ValueError(f"Unknown stage {args.stage}")

if __name__ == "__main__":
    main()