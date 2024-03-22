# memo

## 生成の手順

0. `source ../../../venv/bin/activate`
1. `python synth_tacotron.py`
2. `./synth_pwg.sh`

## synth_tacotron.py

学習済みのpthから、npyファイルを生成する  

1. `conf/train_tacotron/gen_config.yaml`の`inf_dir`を生成したファイルを保存したいディレクトリに変更する。
    - 同時に、`spk_emb_matrix_path`も正しい値に変更する
2. `synth_tacotron.py`内の`pth_dir_path`を使用するexpディレクトリに変更する
3. `pth_list`を必要なエポック分だけ指定する
4. 中間話者音声を生成したい場合、eval.list, spk2id, spksも修正し、対応したfeatsを準備すること
    - 生成するためのconfigも変更する必要がある。
    - チェックリスト
        - [x] eval.list -> all_eval.list として準備完了
        - [x] spk2id -> all_spk2id として準備完了
        - [x] spks -> all_spks として準備完了
        - [x] config -> gen_all_mid_config として準備完了

## synth_pwg.sh

npyファイルからwavを生成する。  

1. `dump_base_dir`を生成したnpyファイルの置き場に変更する

## dir

2024-01-05_14:34:16_d64_baseline
