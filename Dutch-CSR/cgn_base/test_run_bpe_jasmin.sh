#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=1 #0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=2         # number of gpus ("0" uses cpu, otherwise use gpu)
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
tag="cgn_all16k_sp_specaug" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=train_dev
recog_set="g1_q g2_q g3_q g4_q g5_q g1_p g2_p g3_p g4_p g5_p"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; #mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; #mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    mfccdir=`pwd`/mfcc

    for folder in ${recog_set} ; do
    cp -r data/${folder} data/${folder}_novtln
    done

    use_vtln=false
    if $use_vtln; then

        for t in ${recog_set} ; do
            rm -r data/${t}_novtln/{split,.backup,spk2warp} 2>/dev/null || true
            steps/make_mfcc.sh --mfcc-config conf/mfcc_vtln.conf --nj 30 --cmd "$train_cmd" data/${t}_novtln exp/make_mfcc $mfccdir
        done

    for t in ${recog_set} ; do
    local/compute_vad_decision.sh data/${t}_novtln exp/make_mfcc $mfccdir
    done
    
     for t in ${recog_set}; do
        local/get_vtln_warps.sh --nj 4 --cmd "$train_cmd" \
        data/${t}_novtln exp/vtln exp/${t}_warps
        cp exp/${t}_warps/utt2warp data/$t/
     done

    fi

    echo "VTLN Done"

    for rtask in ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${rtask} exp/make_fbank/${rtask} ${fbankdir}
        utils/fix_data_dir.sh data/${rtask}
    done

    for rtask in ${recog_set} ; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}

#INFO CGN:   # In CGN 'ggg' represents laughter sound, 'xxx' represents sounds human unable to recognize
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
echo "dictionary: ${dict}"

nlsyms=data/lang_char/non_lang_syms.txt

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    ###mkdir -p data/lang_char/

    echo "make a non-linguistic symbol list"
    ##cat data/${train_set}/text data/${train_dev}/text data/test_stu/text data/test_tel/text > data/lang_char/alltext
    
    wc -l ${dict}

    # make json labels

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --nj 1 --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done


fi

# you can skip this and remove --rnnlm option in the recognition (stage 5)

if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
lmexpdir=exp/${lmexpname}

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi

expdir=exp/${expname}

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
 
        recog_model=model.last${n_average}.avg.best

    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    #(
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        
        nj=30
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0
        
        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  #\
            #--api v2
            #--rnnlm ${lmexpdir}/${lang_model} \


        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
 
