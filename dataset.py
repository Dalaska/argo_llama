"""
preprocess and serve the argoverse dataset as a DataLoader.
"""
import glob
import os
import random
import torch
import torch.distributed as dist
import pickle


class PrepareDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, input_length, input_width, data_dir):
        super().__init__()
        self.split = split
        self.input_length = input_length
        self.input_width = input_width
        self.data_dir = data_dir

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        DATA_DIR = self.data_dir
        shard_filenames = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))

        # train/test split. let's use only shard 0 for test split, rest train
        val_len = 4000 #1
        shard_filenames = shard_filenames[val_len:] if self.split == "train" else shard_filenames[:val_len]
        assert len(shard_filenames)>0, f"No bin files found in {DATA_DIR}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                #load pkl
                with open(shard, 'rb') as f:
                    x, y = pickle.load(f)
                x = torch.from_numpy(x)
                y = torch.from_numpy(y)
                x = x.float()
                y = y.float()
                yield x, y
 
# -----------------------------------------------------------------------------
# public interface functions
 
class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PrepareDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

