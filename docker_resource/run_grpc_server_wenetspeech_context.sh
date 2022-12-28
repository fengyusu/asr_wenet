
export GLOG_logtostderr=1
export GLOG_v=2
model_dir=/home/docker_resource/wenetspeech/20220506_u2pp_conformer_libtorch
context_path=/home/docker_resource/wenetspeech/20220506_u2pp_conformer_libtorch/context
lm_path=/home/docker_resource/wenetspeech/20220506_u2pp_conformer_libtorch/context/lm
/home/asr_wenet/wenet/wenet/runtime/libtorch/build-grpc-release/bin/grpc_server_main \
    --port 10087 \
    --workers 4 \
    --chunk_size 16 \
    --context_path $context_path \
    --context_score 2.0 \
    --nbest 500 \
    --use_quant_model false \
    --lm_onnx_dir $lm_path \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log







    
