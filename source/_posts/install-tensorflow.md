---
title: 텐서플로우 1.0(최신버전) 설치하는 방법
categories:
  - Tensorflow
  - App
tags:
  - tensorflow
  - install
date: 2017-02-18 18:56:35
thumbnail: https://deeptensorflow.github.io/images/logo.png
---
# 텐서플로우 설치

텐서플로우 1.0버전이 출시되었습니다~
API변동이 약간 있는 것으로 파악되며 앞으로 텐서플로우1.0을 분석하여 블로그를 작성해보도록 하겠습니다.
텐서플로우 공식 홈페이지에서 권장하는 방식인 virtualenv로 설치하는 방법을 택하였습니다.
virtualenv를 사용하면 다른 python 프로그램들에 간섭하거나 영향받는 일이 없어집니다.

## 텐서플로우 MAC 설치

- homebrew 설치

{% codeblock %}
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
{% endcodeblock %}

homebrew는 mac의 패키지 관리자입니다.

- python3 설치

{% codeblock %}
brew install python3
{% endcodeblock %}

- virtualenv 설치

{% codeblock %}
pip3 install virtualenv
{% endcodeblock %}

- virtualenv 환경 세팅(tensorflow 폴더)

{% codeblock %}
virtualenv tensorflow
source tensorflow/bin/activate
cd tensorflow
{% endcodeblock %}

- tensorflow 최신버전 설치(cpu버전)

{% codeblock %}
pip3 install --upgrade tensorflow
{% endcodeblock %}

- tensorflow 최신버전 설치(gpu버전)

{% codeblock %}
pip3 install --upgrade tensorflow-gpu
{% endcodeblock %}

- 설치에 실패한 경우

pip3 버전이 8.1이하인 경우일 것입니다.
{% codeblock %}
brew upgrade python3
{% endcodeblock %}
해주시고 다시 처음부터 진행해 주시기 바랍니다.

- 가상환경을 종료하고 싶은경우

{% codeblock %}
deactivate
{% endcodeblock %}

## 텐서플로우 LINUX 설치

- python3 설치

{% codeblock %}
sudo apt-get install python3
{% endcodeblock %}

- virtualenv 설치

{% codeblock %}
pip3 install virtualenv
{% endcodeblock %}

- virtualenv 환경 세팅(tensorflow 폴더)

{% codeblock %}
virtualenv tensorflow
source tensorflow/bin/activate
cd tensorflow
{% endcodeblock %}

- tensorflow 최신버전 설치(cpu버전)

{% codeblock %}
pip3 install --upgrade tensorflow
{% endcodeblock %}

- tensorflow 최신버전 설치(gpu버전)

{% codeblock %}
pip3 install --upgrade tensorflow-gpu
{% endcodeblock %}

- 설치에 실패한 경우

pip3 버전이 8.1이하인 경우일 것입니다.

{% codeblock %}
sudo apt-get upgrade python3
{% endcodeblock %}

해주시고 다시 처음부터 진행해 주시기 바랍니다.

- 가상환경을 종료하고 싶은경우

{% codeblock %}
deactivate
{% endcodeblock %}

## 텐서플로우 WINDOW 설치

- python 3.5 설치

다른 운영체제와 다르게 3.6버전을 아직 지원하지 않으니 주의 바랍니다.
https://www.python.org/downloads/release/python-352/
위의 링크에서 3.5버전을 다운받으시기 바랍니다.

- tensorflow 최신버전 설치(cpu버전)

{% codeblock %}
pip3 install --upgrade tensorflow
{% endcodeblock %}

- tensorflow 최신버전 설치(gpu버전)

{% codeblock %}
pip3 install --upgrade tensorflow-gpu
{% endcodeblock %}
