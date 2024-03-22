import os
from contextlib import redirect_stdout
from copy import deepcopy
from logging import Logger
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig  # , read_write
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import hydra
from ttslearn.contrib.multispk_util import setup
# from ttslearn.tacotron.frontend.openjtalk import sequence_to_text
from ttslearn.train_util import get_epochs_with_optional_tqdm
from ttslearn.util import load_utt_list, pad_1d

logger: Logger = None


@torch.no_grad()
def eval_model(
    model, utt_ids, in_feats, in_lens, spk_ids, save_dir
):
    # 複数音声の同時生成に非対応のため、分解して生成している
    for idx in tqdm(range(len(in_feats)), desc="Inference batch", leave=False):
        out, out_fine, _, _ = model.inference(in_feats[idx][: in_lens[idx]], spk_ids[idx])
        np.save(save_dir / f"{utt_ids[idx]}-feats",
                out_fine[: len(out)].cpu().data.numpy())  # 必要な長さに切り捨て


def eval_loop(config, device, model, data_loader, logger, pth_name=""):
    phase = "dev"
    save_dir = Path(to_absolute_path(config.data.dev.inf_dir))
    if pth_name != "":
        save_dir = save_dir / pth_name
    os.makedirs(save_dir, exist_ok=True)
    nepochs = 1

    for epoch in get_epochs_with_optional_tqdm(config.tqdm, nepochs):
        model.eval()
        for idx, (utt_ids, in_feats, in_lens, spk_ids,) in tqdm(
            enumerate(data_loader), desc=f"{phase} iter", leave=False, total=len(data_loader)
        ):
            # ミニバッチのソート
            in_lens = in_lens.to(device)
            in_feats = in_feats.to(device)
            spk_ids = spk_ids.to(device)
            eval_model(model, utt_ids, in_feats, in_lens, spk_ids, save_dir)
        break


def collate_fn(batch):
    """Collate function for multi-speaker Tacotron.

    Args
    -----
    batch (list): [(utt_id, input, spk_id), ... ]

    Returns
    -------
    tuple: Batch of inputs, input lengths, spk ids. """
    utt_ids = [x[0] for x in batch]
    xs = [x[1] for x in batch]
    in_lens = [len(x) for x in xs]
    in_max_len = max(in_lens)
    x_batch = torch.stack([torch.from_numpy(pad_1d(x, in_max_len)) for x in xs])
    il_batch = torch.tensor(in_lens, dtype=torch.long)
    spk_ids = torch.tensor([int(x[2]) for x in batch], dtype=torch.long).view(-1, 1)
    return utt_ids, x_batch, il_batch, spk_ids


class MyDataset(Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths (list): List of paths to input files
        out_paths (list): List of paths to output files
        spk_paths (list): List of paths to speaker ID
    """

    def __init__(self, utt_ids, in_paths, spk_paths):
        self.utt_ids = utt_ids
        self.in_paths = in_paths
        self.spk_paths = spk_paths

    def __getitem__(self, idx):
        """Get a pair of input and target

        Args:
            idx (int): index of the pair

        Returns:
            tuple: utt_id, input, speaker ID
        """
        return self.utt_ids[idx], np.load(self.in_paths[idx]), np.load(self.spk_paths[idx])

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_paths)


# config_name = "gen_all_mid_config"
config_name = "gen_config"


@hydra.main(config_path="conf/train_tacotron", config_name=config_name)
def my_app(config: DictConfig) -> None:
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_path = "/home/hosoi/git/ttslearn/extra_recipes/jvs/multispk_tacotron2_pwg/"
    # pth_dir_path = "exp/jvs001-100_sr24000/tacotron2_rf2"
    # pth_dir_path = "exp_002_008_w_audio"
    # pth_dir_path = "exp_wo_spkpre"
    # pth_dir_path = "exp_002_008_w_audio_rate_vo07"
    # pth_dir_path = "exp_002_008_w_audio_rate_vo90"
    # pth_dir_path = "exp_xvec/2023-12-04_05:31:33_d64"
    # pth_dir_path = "exp_xvec/2023-12-22_13:15:47_d64_norm"
    # pth_dir_path = "exp_xvec/2024-01-05_14:34:16_d64_baseline"
    pth_dir_path = "exp_xvec/2024-01-22_03:41:08_baseline_wo_norm"
    pth_list = [
            # "best_loss.pth",
            # "epoch0050.pth",
            "epoch0100.pth",
    ]
    defalut_config = deepcopy(config)
    print("setup")
    # with redirect_stdout(open(os.devnull, 'w')):
    _, _, _, _, _, logger = setup(defalut_config, device, collate_fn)
    in_dir = Path(to_absolute_path(config.data.dev.in_dir))
    utt_ids = load_utt_list(to_absolute_path(config.data.dev.utt_list))
    in_feats_paths = [in_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]
    spk_id_paths = [in_dir / f"{utt_id}-spk.npy" for utt_id in utt_ids]
    for pth_file in tqdm(reversed(pth_list), total=len(pth_list), desc="pth_list", leave=False):
        pth_path_file = os.path.join(project_path, pth_dir_path, pth_file)
        # with read_write(config):
        config.train.pretrained.checkpoint = pth_path_file
        model, _, _, _, _, _ = setup(config, device, collate_fn)

        # リセットしたいので、data_loaderは毎回生成する
        data_loader = DataLoader(
            MyDataset(utt_ids, in_feats_paths, spk_id_paths),
            batch_size=config.data.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=config.data.num_workers,
            shuffle=False,
        )
        eval_loop(config, device, model, data_loader, logger, pth_name=pth_file[:-4])


if __name__ == "__main__":
    my_app()
