export BYTEPS_TRACE_ON=1
export BYTEPS_TRACE_START_STEP=10
export BYTEPS_TRACE_END_STEP=20
export BYTEPS_TRACE_DIR=./traces
export BYTEPS_SERVER_LOG_PATH=./traces/server_log.txt
export BYTEPS_KEY_DICT_PATH=./traces/key_dict.txt
export NVIDIA_VISIBLE_DEVICES=1
export USE_BYTEPS=1

NVIDIA_VISIBLE_DEVICES=1 python3 run_pretraining_local.py --batch_size 32 --max_seq_length 32