#!/usr/bin/env bash
set -eou pipefail

stage=0
stop_stage=4

. shared/parse_options.sh || exit 1

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Stage 0: Organize data"
  # Ensure your data is properly organized
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Prepare manifest"
  python local/prepare_basque_manifest.py
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: Compute fbank features"
  python local/compute_fbank_basque.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Prepare BPE model"
  python local/prepare_bpe_model.py
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Validate prepared data"
  python local/validate_basque_data.py
fi