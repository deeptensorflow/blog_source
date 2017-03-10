---
title: 텐서플로우1.0 설치시 일어나는 기본 오류를 잡아보자 (bazel로 소스코드 빌드)
categories:
  - Tensorflow
  - App
tags:
  - tensorflow
  - install
  - bazel
date: 2017-02-19 12:23:03
thumbnail: https://deeptensorflow.github.io/images/logo.png
---
# 텐서플로우 1.0을 pip으로 설치하고 사용하면 일어나는 오류

tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.

이런 오류를 보신적이 없으신가요?
텐서플로우1.0을 pip(pip3)으로 설치하면 생기는 오류입니다.
간단한 방법으로 (하지만 시간이 조금 걸리는 방법) 해결할 수 있습니다.

# bazel을 설치하자

bazel은 구글에서 만든 빌드 툴입니다.
https://bazel.build/ 에서 다운로드가 가능합니다.
apt-get이나 homebrew 를 통해서도 설치가 가능하구요~
자 설치가 끝났으면 텐서플로우 소스코드를 직접 빌드해보겠습니다.

예)

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


# 텐서플로우 소스코드 빌드하는 방법

- 텐서플로우 소스코드를 깃허브에서 받은 다음에 python3, numpy,  wheel, six를 pip을 통해 받습니다.

{% codeblock %}
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout r1.0 #빌드를 원하는 버전을 입력하시면 되요.
sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel #ubuntu 쓰는 분들만
brew install python3 #맥 쓰는 분들만
sudo pip3 install six numpy wheel #맥쓰는 분들만
{% endcodeblock %}

- GPU 쓰실 분들은 아래의 명령어도 터미널에서 입력해주세요.

저는 PC에 GPU가 안달려있어서 테스트를 못해봤네요.
조만간 GPU를 달 계획입니다!
그런데 아마 오류는 없을거에요.
{% codeblock %}
sudo apt-get install libcupti-dev #ubuntu 쓰는 분들만
brew install coreutils #맥쓰는 분들만
sudo xcode-select -s /Application/Xcode-7.2/Xcode.app #맥쓰는 분들만
{% endcodeblock %}

- configure를 해봅시다.

{% codeblock tensorflow.org code %}
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
{% endcodeblock %}

특별히 신경써야할 부분은 처음에 나오는 이부분들입니다.

{% codeblock %}
Please specify the location of python. [Default is /usr/bin/python]: (설정할 파이썬 path)
{% endcodeblock %}

{% codeblock %}
Do you wish to build TensorFlow with CUDA support? Y(gpu 쓰실 분들은 y해야겠죠~)
{% endcodeblock %}

잘 모르시겠으면 그냥 다 N 누르면서 진행하시면 원래 쓰시던 tensorflow 나올거에요~
영어 잘하시면 직접 해석하시면서 설정을 해주세요. ㅎㅎ

- bazel로 빌드하기

{% codeblock %}
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package #cpu버전일경우
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package  #gpu버전일 경우
{% endcodeblock %}

- 패키지화하기

{% codeblock %}
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
{% endcodeblock %}

- pip 패키지 설치하기

{% codeblock %}
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.0.0-py2-none-any.whl #python2버전
sudo pip3 install /tmp/tensorflow_pkg/tensorflow-1.0.0-cp36-cp36m-macosx_10_12_x86_64.whl #python3버전
{% endcodeblock %}
이건 설정에 따라서 다르닌깐요
sudo pip(파이썬3이면 pip3) install /tmp/tensorflow_pkg/ 한다음에 tab키 쳐주시면 뜨는데 그리고 엔터 눌러주세요.

# 끝!
이제 끝났습니다.
이제 저런 오류없이 텐서플로우가 작동할 것입니다.
긴 글 읽어주셔서 고맙습니다.
https://www.tensorflow.org/install/install_sources 를 참고하여 작성한 문서입니다.
