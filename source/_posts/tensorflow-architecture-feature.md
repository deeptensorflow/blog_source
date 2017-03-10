---
title: 텐서플로우 내부 구조에 대해 알아보자
categories:
  - Tensorflow
  - Internal
tags:
  - tensorflow
  - internal
date: 2017-02-21 15:33:44
thumbnail: https://deeptensorflow.github.io/images/tensorflowarchitecture.png
---
본 내용은 https://www.tensorflow.org/extend/architecture 공홈의 내용을 번역한 것입니다.

# 텐서플로우의 내부는 어떻게 생겼을까?

![tensorflowarchitecture](https://deeptensorflow.github.io/images/tensorflowarchitecture.png)
요런식으로 생겼습니다.
주목해야할 점들은 다음과 같습니다.
- 클라이언트(tensorflow로 코드를 작성하는 부분)
- 계산을 data flow graph으로 정의
- 세션을 사용하여 그래프 실행
- Distributed Master
- Session.run ()의 인수로 사용된 특정부분의 그래프를 정리
- 하위 그래프를 다른 프로세스와 장치에서 실행되는 여러 조각으로 분할
- 그래프 조각을 worker에 뿌림
- worker service
- 사용 가능한 하드웨어 (CPU, GPU 등)에 적합한 커널 구현을 사용하여 그래프 작업 스케줄링 가능
- 다른 작업자 서비스와 작업 결과를 송수신
- 커널 구현
- 개별 그래프에 대한 연산 수행

# 대략적인 워크 플로우 그래프
![텐서플로우워커](https://deeptensorflow.github.io/images/tensorflow_worker.png)
저 위의 그림에서 MASTER는 분산 프로그래밍된 TensorFlow에만 존재합니다.
Tensorflow가 단일버전으로 이루어져있다면 마스터가하는 모든 작업을 수행하지만 로컬 프로세스의 장치와만 통신하는 특수 세션 구현이 포함됩니다.

# 클라이언트에서 일어나는 일
사용자는 계산 그래프를 작성하는 클라이언트 TensorFlow 프로그램을 작성합니다.
여러 라이브러리들을 써서 작업할수도 있으며 여러 layer들을 구성해 가며 추상화 작업을 진행합니다.
TensorFlow는 Python과 C++언어를 지원합니다.

# Distributed master에서 일어나는 일
그래프를 분할하는 작업을 합니다.
이렇게 분할된 그래프에 분산 노드간에 정보를 전달하기 위해 송수신 노드를 삽입합니다
![psandworker](https://deeptensorflow.github.io/images/psandworker)
그리고는 task에게 일을 전달하는 것입니다.
이런식으로 텐서플로우에서는 코드를 병렬적으로 빠르게 수행할 수가 있습니다.

# Worker가 하는 일
- 마스터로부터 온 일을 처리함
- 연산에 대한 커널의 실행들을 스케줄링함
- 작업간의 직접적인 통신 역할(한쪽이 죽으면 그 일을 다른곳에 넘긴다던지..)
Worker는 커널을 로컬 장치에 디스패치하고 가능하면 다중 CPU 코어 또는 GPU 스트림을 사용하여 병렬로 커널을 실행합니다.

즉 Worker는 장치 유형의 각 쌍에 대해 Send 및 Recv 작업을 전문적으로 수행합니다.
1) CPU와 CPU 끼리는 cudaMemcpyAsync() API를 사용하여 계산 및 데이터 전송을 중첩합니다.
2) CPU와 GPU 끼리는 값 비싼 복사를 피하기 위해 peer to peer DMA를 사용합니다.

작업간 전송의 경우 tensorflow는 다음의 프로토콜을 사용합니다.
1) TCP를 통한 gRPC
2) 수렴형 이더넷을 통한 RDMA

또한 다중 GPU 통신을위한 NVIDIA의 NCCL 라이브러리에 대한 예비 지원을 받았습니다.

# 커널 구현
런타임에는 수학, 배열 조작, 제어 흐름 및 상태 관리 작업을 포함하여 200 개가 넘는 표준 작업이 포함됩니다.
각 작업들은 각 device에 맞게 최적화시킬 수 있습니다.
많은 운영 커널은 Eigen :: Tensor를 사용하여 구현되며, C ++ 템플릿을 사용하여 멀티 코어 CPU 및 GPU를위한 효율적인 병렬 코드를 생성합니다.
그러나 우리는보다 효율적인 커널 구현이 가능한 cuDNN과 같은 라이브러리를 자유롭게 사용합니다.
우리는 또한 모바일 장치 및 고처리량 데이터센터 응용프로그램과 같은 환경에서 더 빠른 추론을 가능하게하는 quantization을 구현했으며 quantum 연산을 가속화하기 위해 gemmlowp low-precision matrix library 를 사용합니다.
하위 연산을 연산 조합으로 나타 내기가 어렵거나 비효율적 인 경우 사용자는 C ++로 작성된 효율적인 구현을 제공하는 추가 커널을 등록 할 수 있습니다.
