#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=1 #0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# config files
preprocess_config=conf/specaug.yaml  # use conf/specaug.yaml for data augmentation
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=5

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

vtln_folder=MultiV1
train_dev=""
recog_set=""


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    mfccdir=`pwd`/mfcc

    use_vtln=true
    if $use_vtln; then

    for folder in ${vtln_folder}; do
         cp -r data/${folder} data/${folder}_novtln
    done

    for t in ${vtln_folder}; do
         rm -r data/${t}_novtln/{split,.backup,spk2warp} 2>/dev/null || true
         steps/make_mfcc.sh --mfcc-config conf/mfcc_vtln.conf --nj 30 --cmd "$train_cmd" data/${t}_novtln exp/make_mfcc $mfccdir
    done

    for t in ${vtln_folder}; do
    local/compute_vad_decision.sh data/${t}_novtln exp/make_mfcc $mfccdir
    done
    # Vtln-related things:
    # We'll use a subset of utterances to train the GMM we'll use for VTLN warping.
    num=$(< data/${vtln_folder}/utt2spk wc -l)
    sub=$(($num / 2))
    echo "Total number in segmnets"$num "Choose Half" $sub
    utils/subset_data_dir.sh data/${vtln_folder}_novtln $sub data/${vtln_folder}_sub_novtln
    utils/fix_data_dir.sh data/${vtln_folder}_sub_novtln

    # Note, we're using the speaker-id version of the train_diag_ubm.sh script, which
    # uses double-delta instead of SDC features to train a 256-Gaussian UBM.
    local/train_diag_ubm.sh --nj 30 --cmd "$train_cmd" data/${vtln_folder}_sub_novtln 256 exp/diag_ubm_vtln


    local/train_lvtln_model.sh --mfcc-config conf/mfcc_vtln.conf --nj 30 --cmd "$train_cmd" data/${vtln_folder}_sub_novtln exp/diag_ubm_vtln exp/vtln_${vtln_folder}

     for t in ${vtln_folder}; do
        local/get_vtln_warps.sh --nj 4 --cmd "$train_cmd" \
        data/${t}_novtln exp/vtln_${vtln_folder} exp/${t}_warps
        cp exp/${t}_warps/utt2warp data/$t/
     done

    echo "VTLN Done"
    fi

fi

