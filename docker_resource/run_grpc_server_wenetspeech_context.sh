
export GLOG_logtostderr=1
export GLOG_v=2
model_dir=/home/asr/docker_resource/wenetspeech/20220506_u2pp_conformer_libtorch
context_path=/home/asr/docker_resource/wenetspeech/20220506_u2pp_conformer_libtorch/context
/home/asr/asr-wenet/wenet/wenet/runtime/libtorch/build-grpc-release/bin/grpc_server_main \
    --port 10087 \
    --workers 4 \
    --chunk_size 16 \
    --context_path $context_path \
    --context_score 2.0 \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log
    
