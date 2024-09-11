MAIN_ROOT=/tudelft/SpeechLab/Software/espnet-v8/espnet

export KALDI_ROOT=/tudelft/SpeechLab//Software/kaldi
export FST_ROOT=/tudelft/SpeechLab/Software/kaldi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
#export PATH=$PWD/utils/:$FST_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=$KALDI_ROOT/tools/portaudio/lib:$KALDI_ROOT/tools/openfst-1.7.2/lib:$KALDI_ROOT/src/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8


