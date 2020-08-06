#!/bin/bash
# set -x
if [ $# -lt 3 ]; then
    echo "usage: $0 rank num_servers num_workers"
    exit -1;
fi

export RANK=$1
export NUM_SERVERS=$2
export NUM_WORKERS=$3
echo "RANK= ${RANK}"
echo "NUM_SERVERS = ${NUM_SERVERS}"
echo "NUM_WORKERS = ${NUM_WORKERS}"

export BYTEPS_TRACE_ON=1
export BYTEPS_TRACE_START_STEP=10
export BYTEPS_TRACE_END_STEP=20
export BYTEPS_TRACE_DIR=./traces
export BYTEPS_SERVER_LOG_PATH=./traces/server_log.txt
export BYTEPS_KEY_DICT_PATH=./traces/key_dict.txt
export NVIDIA_VISIBLE_DEVICES=1
export DMLC_WORKER_ID=${RANK}
export DMLC_NUM_WORKER=${NUM_WORKERS}
export DMLC_NUM_SERVER=${NUM_SERVERS}
export BYTEPS_RANK=${RANK}
export BYTEPS_SIZE=${NUM_WORKERS}
export USE_BYTEPS=1

# start the scheduler
export DMLC_PS_ROOT_URI='10.28.1.18'
export DMLC_PS_ROOT_PORT=56723

if [ ${RANK} -eq 0 ]
then
    echo "Starting scheduler on rank 0."
    export DMLC_ROLE='scheduler'
    ./run_bert_pretrain.sh &
fi


# start server
export DMLC_NODE_HOST='10.28.1.18'
export BYTEPS_SERVER_ENABLE_PROFILE=1
export BYTEPS_SERVER_PROFILE_OUTPUT_PATH=./traces/server_profile.json
export DMLC_ROLE='server'
export PORT='56724'
export HEAPPROFILE=./S${RANK}
./run_bert_pretrain.sh &

# start worker
export DMLC_ROLE='worker'
export PORT='56725'
export HEAPPROFILE=./W${RANK}
./run_bert_pretrain.sh &

wait
