---
title: 텐서플로우 빌드하는 방법 (텐서플로우에 내코드 추가하기)
categories:
  - Tensorflow
  - Internal
tags:
  - tensorflow
  - internal
  - build
date: 2017-02-21 09:47:00
thumbnail: https://deeptensorflow.github.io/images/bazel.png
---
# 직접 bazel로 빌드했을때의 장점?
tensorflow 커뮤니티에 올라온 정보들에 따르면 직접 빌드했을때 속도가 소폭 향상된다고 합니다.
추후 직접 실험을 해서 속도비교를 해서 올릴 예정입니다.
그리고 자신이 직접 내부의 소스코드를 수정해서 빌드하면 '나만의 텐서플로우'를 만들 수 있습니다.

# 빌드 전 개발환경 세팅
pip(파이썬 패키지 매니저)과 java-jdk가 설치되어 있어야 합니다.
설치 방법은 OS환경마다 다르지만 매우 간단합니다.
아마 대부분 설치가 이미 되어있을 것이라고 생각하기 때문에 그냥 넘어가겠습니다.

# bazel 다운로드
https://bazel.build/ 에서 다운로드가 가능합니다.
curl로 다운 받는 방법은 아래와 같습니다.
```
export BAZELRC=/home/<yourid>/.bazelrc
export BAZEL_VERSION=0.4.2
mkdir /home/<yourid>/bazel
cd /home/<yourid>/bazel
curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
curl -fSsL -o /home/<yourid>/bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE.txt
chmod +x bazel-*.sh
sudo ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
cd /home/<yourid>/
rm -f /home/<yourid>/bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
```

# tensorflow github에서 다운로드
소스가 있어야 빌드를 하겠죠?
그리고 wheel, six, numpy 패키지가 필요합니다.
```
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout r1.0 #빌드를 원하는 버전을 입력하시면 되요.
sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel #ubuntu 쓰는 분들만
brew install python3 #맥 쓰는 분들만
sudo pip3 install six numpy wheel #맥쓰는 분들만
```
여기서 빌드를 하기 전에 소스코드를 고쳐서 텐서플로우에 내 소스를 추가할 수 있습니다.
공식 홈페이지에서 해당 내용도 있습니다.
https://www.tensorflow.org/extend/adding_an_op
다음 포스팅으로 자세히 해당 내용에 대해서도 다루어 보겠습니다.

# tensorflow에서는 설정을 쉽게 하는 방법을 제공합니다.
tensorflow 폴더에서 ./configure 하시면 빌드 설정을 쉽게 하도록 도와줍니다.
```
$ cd tensorflow  # cd to the top-level directory created
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python2.7
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
Do you wish to use jemalloc as the malloc implementation? [Y/n]
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N]
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N]
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N]
No XLA JIT support will be enabled for TensorFlow
Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]
Using python library path: /usr/local/lib/python2.7/dist-packages
Do you wish to build TensorFlow with OpenCL support? [y/N] N
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] Y
CUDA support will be enabled for TensorFlow
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 8.0
Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify the cuDNN version you want to use. [Leave empty to use system default]: 5
Please specify the location where cuDNN 5 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 3.0
Setting up Cuda include
Setting up Cuda lib
Setting up Cuda bin
Setting up Cuda nvvm
Setting up CUPTI include
Setting up CUPTI lib64
Configuration finished
```
중간에 파이썬 버전 선택이나 CUDA 지원 여부 자신의 환경에 맞게 잘 세팅해주세요.
제안들이 친절하게 분류되어있어서 어려움은 없을 것 같습니다.

# bazel로 빌드하기
설정이 끝났다면 빌드를 해야겠죠?
```
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package #cpu버전일경우
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package  #gpu버전일 경우
```

pip 패키지로 관리하기 쉽게 만들어줍시다.
```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

pip으로 다운받아줍시다.
```
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.0.0-py2-none-any.whl #python2버전
sudo pip3 install /tmp/tensorflow_pkg/tensorflow-1.0.0-cp36-cp36m-macosx_10_12_x86_64.whl #python3버전
```
이건 설정에 따라서 다르닌깐요
sudo pip(파이썬3이면 pip3) install /tmp/tensorflow_pkg/ 한다음에 tab키 쳐주시면 뜨는데 그리고 엔터 눌러주세요.
