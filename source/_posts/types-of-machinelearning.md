---
title: 머신러닝에는 어떤 종류의 학습들이 있을까요?
categories:
  - Machinelearning
  - News
tags:
  - machinelearning
date: 2017-03-10 13:44:19
thumbnail: https://deeptensorflow.github.io/images/ml_algorithm.png
---
# 머신러닝에서 학습의 종류

- 지도학습

트레이닝 세트와 각각에 대한 목표값이 제공되고 이 제공된 데이터세트를 통해 모든 입력 값에 대해 정답을 유추해 낼 수 있도록 일반화시킵니다.
이것은 또한 예제를 통한 학습이라고도 불립니다(주로 함수 근사법, 또는 보간법이 쓰입니다).

- 비지도학습

정답이나 목표값이 제공되지 않는 경우 알고리즘이 정답을 제공한다기보다는 예제간에 유사점을 찾아내고, 이를 통해서 분류 체계를 확립할 수 있게 합니다.
통계학적인 비지도학습은 밀도추정이라고 불립니다.

- 강화학습

지도학습과 비지도학습의 중간쯤에 속합니다.
알고리즘이 출력하는 결과 값에 대해 어느 답이 틀렸는지를 알려주지만, 이를 어떻게 고쳐야 하는지는 알려주지 않습니다.
그래서 알고리즘이 직접 다양한 가능성을 스스로 정답을 얻을 때까지 시도해 보도록 만듭니다.
강화학습은 시험지에 작성된 답에 점수를 매기지만, 어떻게 발전시킬 수 있는지를 말해주지 않으므로 비판을 통한 학습이라고도 불립니다.

- 진화학습

생물학적 진화 역시 학습 과정이라고 여겨질 수 있는데, 생물학적인 유기체는 살아남을 확률을 높이고, 그들의 후손을 남기기 위해서 환경에 적응합니다.
어떻게 이 진화학습 모델들이 컴퓨터에 적용되는지는 적합도를 통해서 살펴볼 것입니다.
