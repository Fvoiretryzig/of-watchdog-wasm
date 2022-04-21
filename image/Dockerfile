# FROM --platform=${TARGETPLATFORM:-linux/arm64} alpine:3.14 as build
FROM nvcr.io/nvidia/l4t-base:r34.1
# FROM --platform=${TARGETPLATFORM:-linux/arm64} nvidia/cuda:11.0.3-runtime-ubuntu18.04 as build

RUN apt-get update && apt install -y libncurses5 gcc
COPY libwasmer.so /usr/lib
# COPY go1.16.13.linux-arm64.tar.gz /
# RUN tar -C /usr/local -xzf go1.16.13.linux-arm64.tar.gz \
# 	&& mkdir -p /go/src	\
COPY fwatchdog-arm64 /usr/bin/fwatchdog

# RUN addgroup --system app && adduser --system --ingroup app app \
#    && mkdir -p /home/app \
#    && chown app /home/app

#WORKDIR /home/app
COPY wasm_root/ /wasm_root

#RUN chown -R app /home/app
#USER app

# ENV fprocess="mulMatrix.wasm 1024"
ENV fprocess="device.wasm"
ENV mode=wasm
ENV wasm_root=/wasm_root

HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1
CMD ["fwatchdog"]

