FROM asr_wenet_raw:latest
MAINTAINER <fengyu@stu.pku.edu.cn>
ENV DEBIAN_FRONTEND=noninteractive

# RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
# RUN apt-get update && apt-get install -y git cmake wget build-essential

# RUN export GIT_TRACE_PACKET=1
# RUN export GIT_TRACE=1
# RUN export GIT_CURL_VERBOSE=1
# RUN git clone https://github.com/fengyusu/asr_wenet.git /home/asr_wenet

# ARG model=20210618_u2pp_conformer_libtorch.tar.gz
# RUN wget -P /home https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/$model
# RUN tar -xzf /home/$model -C /home

ARG build=/home/asr_wenet/wenet/wenet/runtime/libtorch/build-grpc-release
RUN mkdir $build && cd $build && cmake -DGRPC=ON .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release

