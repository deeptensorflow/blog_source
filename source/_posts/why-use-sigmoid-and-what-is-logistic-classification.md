---
title: 시그모이드 함수는 왜 쓰일까? logistic regression, softmax regression
categories:
  - Machinelearning
  - News
tags:
  - machinelearning
  - algorithm
date: 2017-02-19 18:08:56
thumbnail: https://deeptensorflow.github.io/images/sigmoid.png
---

# 시그모이드 함수란?

![sigmoid](https://deeptensorflow.github.io/images/sigmoid.png)
그림 실력 지못미 ㅠㅠ
위의 그림처럼 생긴 함수입니다.
0과 1 사이에서 값이 정해진다는 것만 유심히 보시면 됩니다.
linear regression을 지난번 포스팅에서 y = W * X + b의 꼴로 설명드렸지요?
(https://deeptensorflow.github.io/2017/02/19/what-is-linear-regression-with-tensorflow/)
이러한 꼴로는 해석하기 힘든 현상들이 있어서 나온 것이 바로 logistic regression이고 여기서 쓰이는 함수가 sigmoid 함수입니다.

# logistic regression이 나오게 된 계기를 먼저 알자
이전 포스팅에서 다룬 linear regression을 적용하기 힘든 부분은 바로 true of flase 꼴을 처리할 때입니다.
왜 처리하기가 힘드냐? 기본적으로 y = a*x + b 꼴의 선형 함수는 우선 x에 따른 대응되는 y값이 무한대입니다.
0과 1로 처리를 하고싶은 logistic classification의 경우 적용하기 무리가 있지요.
0 아니면 1, true 아니면 flase, 모 아니면 도 이런 경우이지요.
현실의 예를 들자면 스팸메일인지 아닌지, 합격인지 불합격인지가 이런 경우에 해당됩니다.
그래서 이를 대체할 함수를 찾다가! 딱! 시그모이드 함수를 발견하게 된 것입니다.
그리고 시그모이드 함수를 기존의 linear regression에 그대로 씌워봤는데 효과가 좋아서 이것을 logistic regression이라고 하게 된 것이지요.
기본적으로 0과 1사이에서 값이 노는데다가 0.5 기준점으로 원점대칭 꼴이라서 코드 적용도 쉽습니다.
기존의 linear regression a*x+b의 부분을 시그모이드 함수의 x부분에 넣어주었더니 실제로 양자택일의 경우에 굉장히 뛰어난 효율을 보여줬습니다.(1/(1+e^(-(ax+b))))
현재도 정말 유용하게 쓰이고 있는 함수 모델입니다.

# 그냥 바로 코드에 적용시켜도 되나요?
sigmoid 를 이용한 logistic classification 코드를 작성하는 방법을 설명해주는 강의가 있습니다.
{% youtube 6vzchGYEJBc %}
요약하자면 cost함수를 구성할때 그대로 적용하면 굴곡진 부분들이 많아져서 gradient descent 알고리즘 적용에는 무리가 있으니 log함수를 취해줘서 그래프를 펴주자... 이런 내용입니다.
꼭 한번 강의를 듣는 것을 추천드립니다.
홍콩 과기대의 김성훈 교수님의 강의입니다.

# softmax regression 이라는 것도 있던데...
이것은 그냥 logistic regression을 muti-variable 형식으로 구성한 것이라고 보면 됩니다.
예를 들어 표현해 보도록 하겠습니다.
기존의 logistic regression은 양자택일을 해주었죠?
그리고 그것을 돕는 함수가 sigmoid 함수였습니다.
이것을 한번 응용해 보도록 하겠습니다.
A학점 B학점 C학점이 있다고 가정하겠습니다.
공부하는 시간에 따른 학점이 얼마나 나오는지 알아보고 싶습니다.
공부하는 시간을 X라고 두면 이에 따른 weight와 bias가 있을 것입니다.
대충 Y = W * X + b 꼴이 될 것입니다.
여기에 우리가 배운 logistic regression을 적용하면 시그모이드가 씌워져서 0~1사이의 값이 나오겠죠?
이게 A학점, B학점, C학점에 대해 각각 따로따로 있는 것입니다.
![softmax](https://deeptensorflow.github.io/images/softmax.png)
각 학점에 대한 bias와 weight를 둡니다.
그리고 이것을 sigmoid를 씌워서 0~1사이의 값으로 나오게 하고 각 값들의 합을 1로 하여 확률로 변환하는 과정 까지를 softmax regression이라고 합니다.
그리고 cost함수를 짜겠죠?(log함수를 이용합니다. 위의 유투브 영상에 짜는 방법 나옴!)
실제 값을 출력할 때에는 argmax를 이용해서 가장 확률이 높은 값을 뽑습니다.
