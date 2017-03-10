---
title: 선형회귀란 무엇을까? 텐서플로우를 통해 알아보자(linear regression)
categories:
  - Tensorflow
  - App
tags:
  - tensorflow
  - linearregression
date: 2017-02-19 13:44:28
thumbnail: https://deeptensorflow.github.io/images/linear_regression.png
---
# 선형회귀(linear regression)

- 위키백과

선형회귀란  종속 변수 y와 한 개 이상의 독립 변수 (또는 설명 변수) X와의 선형 상관 관계를 모델링하는 회귀분석 기법입니다.
한 개의 설명 변수에 기반한 경우에는 단순 선형 회귀, 둘 이상의 설명 변수에 기반한 경우에는 다중 선형 회귀라고 합니다.

- 쉽게 설명하자면...

어떤사람은 2시간 공부해서 시험점수를 30점을 받았습니다.
어떤사람은 3시간 공부해서 40점을, 어떤사람은 4시간 공부해서 50점을 받았습니다.
이럴 경우에 공부시간과 시험점수 사이에 어떤 관계가 있는지 알아보고 싶겠죠?
이럴 때 자주 쓰이는 기법이 선형 회귀입니다.
그리고 더 세분화 하자면 결과에 미치는 요인이 한개이기 때문에 단순 선형 회귀라고 부릅니다.
단순한 공부시간 뿐만 아니라 학교의 출석율도 한번 설명 변수로 넣어보겠습니다.
어떤 사람은 2시간을 공부하고 학교에 5번 출석을 해서 20점을 받았습니다.
어떤 사람은 2시간을 공부하고 학교에 10번을 출석해서 30점을 받았구요!
어떤 사람은 5시간을 공부하고 학교에 20번을 출석해서 50점을 받았다고 가정하겠습니다.
공부시간, 학교 출석율이 시험점수에 얼마나 영향을 미치는지 궁금하죠?
이럴때 자주 쓰이는 기법이 선형 회귀입니다.
더 세분화 하자면 설명 변수가 둘 이상이니 다중 선형 회귀라고 합니다.

- 방정식으로 알아보자

위의 경우에 아래의 방정식이 세워지죠?
Y(시험점수) = x1(공부한시간) * w1(공부시간과 시험점수의 상관계수) + x2(학교 출석율) * w2(학교 출석율과 시험점수의 상관계수) + b(보정값)
요약하자면 Y = x1*w1 +x2*w2 + b 입니다.
중학교 때 많이 본 함수이지요?
이런 모델을 써서 설명변수(x1, x2)와 종속변수(Y)의 상관관계를 알고자 하는 방법인 것입니다.

- 텐서플로우로 단순 선형 회귀에 대해 알아보자

일부러 상관관계를 알기 쉽도록 눈에 보이는 모델을 써봤습니다.
공부시간이 1, 2, 3시간일때 시험 점수가 2, 4, 6점이라면?
Y = x1*w1 + b의 모델이 짜질 것입니다.
그리고 w1은 2, b는 0이 되어야 주어진 방정식을 만족할 것이라는 것을 우리는 알 수 있죠.
한번 텐서플로우 코드로 확인해 보겠습니다.

{% codeblock tf_linear_regression %}
# 텐서플로우를 사용할 것을 알려줍니다.
import tensorflow as tf

# x_data는 공부시간, y_data는 시험성적 입니다.
x_data = [1, 2, 3]
y_data = [2, 4, 6]

# W는 설명변수, b는 보정값 입니다.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# placeholder를 사용하면 변수의 형태만 지정해주고 나중에 값을 넣어줘도 됩니다.
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 방정식 모델입니다.
hypothesis = W * X + b

# cost 함수입니다. 뒤어서 설명하겠습니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent 또한 뒤에서 설명하겠습니다.
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# 변수들을 초기화합니다.
init = tf.global_variables_initializer()

# 텐서플로우를 시작하게 하는 구문이라고 보시면 됩니다. 세션을 지정해줍니다.
sess = tf.Session()
sess.run(init)

# X에 x_data를, Y에 y_data를 넣어서 2001번 소스를 돌려가며 W와 b값을 찾아갑니다.
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

# 5시간 공부했을때와 2.5시간 공부했을 때 몇점이 나올지 출력해봅니다.
print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))

{% endcodeblock %}

- 결과입니다.

{% codeblock %}
# 순서대로 몇번째 트레이닝인지, 오차범위, W값, b값 입니다.
0 0.245183 [ 2.28380156] [-0.13001066]
200 3.87879e-07 [ 2.00072336] [-0.00164423]
400 2.25526e-11 [ 2.00000548] [ -1.27652002e-05]
600 1.51582e-13 [ 2.00000072] [ -1.24958285e-06]
800 1.51582e-13 [ 2.00000072] [ -1.24958285e-06]
1000 1.51582e-13 [ 2.00000072] [ -1.24958285e-06]
1200 1.51582e-13 [ 2.00000072] [ -1.24958285e-06]
1400 1.51582e-13 [ 2.00000072] [ -1.24958285e-06]
1600 1.51582e-13 [ 2.00000072] [ -1.24958285e-06]
1800 1.51582e-13 [ 2.00000072] [ -1.24958285e-06]
2000 1.51582e-13 [ 2.00000072] [ -1.24958285e-06]

# 5시간 공부했을 때와 2.5시간 공부했을 때를 출력해줍니다.
[ 10.00000286]
[ 5.00000048]
{% endcodeblock %}

- cost 함수 & gradient descent란?

김성훈 교수님의 강의가 이것을 이해하는데는 최고 같습니다.
아래의 비디오 두편만 보면 이해가 빡!
{% youtube Hax03rCn3UI %}
{% youtube TxIVr-nk1so %}

- 다중 선형 회귀 함수는 어떻게 처리를 할까요?

다중 선형 회귀함수를 보기 좋게 처리하려면 한가지 아이디어가 필요합니다.
행렬의 아이디어가 필요한데요.
Y = X1*W1 + X2*W2 + X3*W3 + .... + b 이런 꼴이 선형 회귀 모델인데요.
이것을 어떻게 간결하게 표현할 방법이 없을까요?
있습니다!
Y = X1*W1 + X2*W2 + X3*W3 +... + 1*b의 꼴로 두면 됩니다.
그리고 이것을 행렬로 변환!
![linearregression](https://deeptensorflow.github.io/images/linear_regression.png)
요런식으로 행렬로 관리하면 되겠죠?
(아, b1, b2, ... bn은 전부 같은 값입니다.)
그런데 더 간략하게 하고 싶다면 b까지 행렬에 포함시킬 수 있습니다.
![multi_linearregression](https://deeptensorflow.github.io/images/multi_linear_regression.png)
이럴경우에 코드는 어떻게 되냐구요?

- 다중 선형 회귀 함수 텐서플로우 예시 코드

{% codeblock %}
import tensorflow as tf

x_data = [[1., 2., 5.],[1., 3., 7.], [1., 4., 10.], [1., 7., 12.]]
y_data = [1., 2., 3., 4.]

W = tf.Variable(tf.random_uniform([3,1], -1, 1))

# 이거 한줄로 가설함수 끝!
hypothesis = tf.matmul(x_data, W)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.01)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)  # goal is minimize cost

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 200 == 0:
        print (step, sess.run(cost), sess.run(W))
{% endcodeblock %}

다중 선형 회귀를 이용했는데도(W 2개 b 1개) hypothesis 함수 코드가 엄청 간결하죠?
행렬의 위력입니다~ (다들 선형대수 열공을..)
참고로 x_data와 y_data에는 별다른 의미 없는 값을 넣었습니다.
그냥 소스코드 참고용이라... ㅎ

# 결론은?
linear regression이 대략적으로 어떤 요인에 미치는 변수들의 패턴을 선형적으로 파악하고자 하는 것이고 tensorflow를 이용하면 쉽게 구현이 가능하다!
