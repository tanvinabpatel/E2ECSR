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

# sample filtering
min_io_delta=4  # samples with `len(input) - len(output) * min_io_ratio < min_io_delta` will be removed.

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

train_set=train_allcomp_sp
train_dev=train_dev
recog_set="test_stu test_tel"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/hkust_data_prep.sh ${hkust1} ${hkust2}
    local/hkust_format_data.sh
    # upsample audio from 8k to 16k to make a recipe consistent with others
    for x in train dev; do
        sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
    done
    # remove space in text
    for x in train dev; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
        rm data/${x}/text.org
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    mfccdir=`pwd`/mfcc

    echo "removelongshort" "manually did it "
    #remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_orig data/${train_set}_reduced
    #remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_orig data/${train_dev}
    
    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 data/train_allcomp data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/train_allcomp data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/train_allcomp data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3

    use_vtln=false
    if $use_vtln; then

    for folder in ${train_set} ${recog_set} ${train_dev}; do
         cp -r data/${folder} data/${folder}_novtln
    done

    for t in ${train_set} ${recog_set} ${train_dev}; do
         rm -r data/${t}_novtln/{split,.backup,spk2warp} 2>/dev/null || true
         steps/make_mfcc.sh --mfcc-config conf/mfcc_vtln.conf --nj 30 --cmd "$train_cmd" data/${t}_novtln exp/make_mfcc $mfccdir
    done


    for t in ${train_set} ${recog_set} ${train_dev}; do
    local/compute_vad_decision.sh data/${t}_novtln exp/make_mfcc $mfccdir
    done
    # Vtln-related things:
    # We'll use a subset of utterances to train the GMM we'll use for VTLN warping.
    num=$(< data/jasmin_all/text wc -l)
    sub=$(($num / 2))
    echo "Total number in segmnets"$num "Choose Half" $sub
    utils/subset_data_dir.sh data/jasmin_all_novtln $sub data/jasmin_all_sub_novtln
    utils/fix_data_dir.sh data/jasmin_all_sub_novtln

    # Note, we're using the speaker-id version of the train_diag_ubm.sh script, which
    # uses double-delta instead of SDC features to train a 256-Gaussian UBM.
    local/train_diag_ubm.sh --nj 30 --cmd "$train_cmd" data/jasmin_all_sub_novtln 256 exp/diag_ubm_vtln


    local/train_lvtln_model.sh --mfcc-config conf/mfcc_vtln.conf --nj 30 --cmd "$train_cmd" data/jasmin_all_sub_novtln exp/diag_ubm_vtln exp/vtln

     for t in ${train_set} ${recog_set} ${train_dev}; do
        local/get_vtln_warps.sh --nj 4 --cmd "$train_cmd" \
        data/${t}_novtln exp/vtln exp/${t}_warps
        cp exp/${t}_warps/utt2warp data/$t/
     done

    echo "VTLN Done"
    fi

    
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 40 --write_utt2num_frames true \
        data/${train_set} exp/make_fbank/${train_set} ${fbankdir}

    utils/fix_data_dir.sh data/${train_set}

    for rtask in ${recog_set} ${train_dev}; do
        steps/make_fbank_pitch.sh  --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${rtask} exp/make_fbank/${rtask} ${fbankdir}
        utils/fix_data_dir.sh data/${rtask}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    split_dir=$(echo $PWD | awk -F "/" '{print $NF "/" $(NF-1)}')
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/a{11,12,13,14}/${USER}/espnet-data/egs/${split_dir}/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/a{11,12,13,14}/${USER}/espnet-data/egs/${split_dir}/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 40 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}

    for rtask in ${train_dev} ${recog_set}; do
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

#dict=data/lang_char/${train_set}_units.txt
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
echo "dictionary: ${dict}"

nlsyms=data/lang_char/non_lang_syms.txt

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    mkdir -p data/lang_char/

    echo "make a non-linguistic symbol list"
    cat data/${train_set}/text data/${train_dev}/text data/test_stu/text data/test_tel/text > data/lang_char/alltext_noutf8
    
    iconv -f latin1 -t UTF-8 data/lang_char/alltext_noutf8 > data/lang_char/alltext
    cut -f 2- -d" " data/lang_char/alltext > data/lang_char/input_${train_set}.txt
    cut -f 2- data/${train_set}/text | grep -o -P '\[.*?\]' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    spm_train --user_defined_symbols=[UNK],[FIL],[LAUGH] --input=data/lang_char/input_${train_set}.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=1000000  --shuffle_input_sentence=true
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input_${train_set}.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l $dict
    wc -l ${dict}


        # make json labels
    data2json.sh --nj 1 --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json


    data2json.sh --nj 1 --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --nj 1 --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done


fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
# Not using LM
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    # use external data
    #if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
    #    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
    #fi
    if [ ! -e ${lmdatadir} ]; then
        mkdir -p ${lmdatadir}
        #iconv -f latin1 -t UTF-8 data/${train_set}/text > data/${train_set}/train_text.txt
        #cut -f 2- -d" " data/${train_set}/train_text.txt | 
        spm_encode --model=${bpemodel}.model --output_format=piece < data/${train_set}/train_text.txt  > ${lmdatadir}/train.txt
        # combine external text and transcriptions and shuffle them with seed 777
        #zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/${train_set}_text.gz |\
        #spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
       # iconv -f latin1 -t UTF-8 data/${train_dev}/text > data/${train_dev}/dev_text.txt
        #cut -f 2- -d" " data/${train_dev}/dev_text.txt | gzip -c > data/local/lm_train/${train_dev}_text.gz 
        spm_encode --model=${bpemodel}.model --output_format=piece < data/${train_dev}/dev_text.txt > ${lmdatadir}/valid.txt
    fi

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict} \
        --dump-hdf5-path ${lmdatadir}
fi

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
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        
# Average ASR models
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
                    --snapshots ${expdir}/results/snapshot.ep.* \
                    --out ${expdir}/results/${recog_model} \
                    --num ${n_average}
    fi

    pids=() # initialize pids
    for rtask in ${recog_set} train_dev; do
    #(
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})
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
            --model ${expdir}/results/${recog_model}  \
            --api v2
            #--rnnlm ${lmexpdir}/${lang_model} \
            #--api v2

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    #) &
    #pids+=($!) # store background pids
    done
    echo "Finished"
fi
 
