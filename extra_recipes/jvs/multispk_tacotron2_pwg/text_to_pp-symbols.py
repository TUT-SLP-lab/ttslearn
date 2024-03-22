import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from os import makedirs
from pathlib import Path

import numpy as np
from pyopenjtalk import extract_fullcontext as ef
from tqdm import tqdm

from ttslearn.tacotron.frontend.openjtalk import pp_symbols, text_to_sequence

# preprocess.pyを参考に作成 (p.310)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Process for text to pp-symbols",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("id_text_list", type=str, help="utterance list")
    parser.add_argument("spk2id", type=str, help="out directory")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    return parser


def preprocess(id: str, text: str, out_dir: Path, spk2id: dict):
    spk = id.split("_")[0]
    ef_text = ef(text)
    symbols = pp_symbols(ef_text)
    feats = np.array(text_to_sequence(symbols), dtype=np.int64)
    np.save(out_dir / f"{id}-symbols.npy", np.array(symbols))
    np.save(out_dir / f"{id}-feats.npy", feats)
    np.save(out_dir / f"{id}-spk.npy", spk2id[spk])


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    out_dir = Path(args.out_dir)
    makedirs(out_dir, exist_ok=True)

    utt_ids = []
    utt_texts = []
    print("gen ids & texts")
    with open(args.id_text_list) as f:
        for line in f:
            _id, _text = line.split(":")
            utt_ids.append(_id)
            utt_texts.append(_text)
    print("gen spk2id")
    spk2id = {}
    with open(args.spk2id) as f:
        for line in f:
            _id, _text = line.split(":")
            spk2id[_id] = np.array([int(_text)])

    with ProcessPoolExecutor(args.n_jobs) as executor:
        print("generate executor futures")
        futures = [executor.submit(preprocess, id, text, out_dir, spk2id) for id, text in zip(tqdm(utt_ids), utt_texts)]
        print("generate text feature")
        for future in tqdm(futures):
            future.result()
