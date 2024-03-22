#!/bin/bash

set -e
set -u
set -o pipefail

function xrun () {
    set -x
    "$@"
    set +x
}

bash_src_dir=$(dirname "${BASH_SOURCE:-$0}")
script_dir=$(cd "$bash_src_dir"; pwd)
echo "$script_dir"
COMMON_ROOT=../../../recipes/common
. ${COMMON_ROOT}/yaml_parser.sh || exit 1;

eval "$(parse_yaml './config.yaml' '')"

vocoder_model=$(basename "${parallel_wavegan_config}")
vocoder_model=${vocoder_model%.*}
# exp name
if [ -z "${tag:=}" ]; then
    expname="jvs001-100_sr${sample_rate}"
else
    expname="jvs001-100_sr${sample_rate}_${tag}"
fi
expdir=exp/$expname


_base_dir="outputs/xvec/2024-01-23_15:08:03_d64_norm"


# _base_dir="outputs/xvec/mid_all/simple/2024-01-23_14:18:28_d64_baseline_wo_norm"
# _base_dir="outputs/xvec/2024-01-23_14:10:48_d64_baseline_wo_norm"
# _base_dir="outputs/xvec/mid_all/simple/2024-01-09_14:28:05_d64_baseline"
# _base_dir="outputs/xvec/2024-01-09_14:15:31_d64_baseline"
# _base_dir="outputs/xvec/mid_all/simple/2023-12-26_16:54:18_d64_norm"
# _base_dir="outputs/xvec/2023-12-26_16:33:21_d64_norm"
# _base_dir="outputs/xvec/mid_all/simple/2023-12-10_18:51:12_64"
# _base_dir="outputs/xvec/2023-11-16_16:33:21"
# _base_dir="outputs/jvs/exp_002_008_w_audio_rate_vo90"
# dump_base_dir="outputs/jvs/spk-rec-2-model/melspec"
# dump_base_dir="outputs/jvs/exp_002_008_w_audio/melspec"
# dump_base_dir="outputs/jvs/normal/melspec"
dump_base_dir="${_base_dir}/melspec"
vocoder_eval_checkpoint=$(find "${expdir}/${vocoder_model}"/*.pkl -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2 || true)
echo "$vocoder_eval_checkpoint"
# outdir="outputs/jvs/exp_002_008_w_audio/wav"
# outdir="outputs/jvs/normal/wav"
outdir="${_base_dir}/wav"

for dumpdir in "${dump_base_dir}"/*; do
	dir_name=$(basename "${dumpdir}")
    xrun parallel-wavegan-decode \
        --dumpdir "${dumpdir}" \
        --checkpoint "${vocoder_eval_checkpoint}" \
        --outdir "${outdir}/${dir_name}"
done
