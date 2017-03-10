---
title: 텐서플로우 맛보기(기본 가이드)
categories:
  - Tensorflow
  - App
tags:
  - tensorflow
date: 2017-02-27 08:35:14
thumbnail: https://deeptensorflow.github.io/images/logo.png
---
이 문서는 https://www.tensorflow.org/get_started/get_started 의 내용을 번역했습니다.

# 텐서플로우를 하기 전에 알아야 할것
1. 파이썬 프로그래밍
2. 배열에 관한 내용 (조금이라도)
3. 머신러닝에 관한 내용 (조금이라도)

참고로 파이썬은 https://wikidocs.net/book/1 이곳에서 공부하시는 것을 추천드립니다.
TensorFlow는 여러 API를 제공합니다.
최저 수준의 API 인 TensorFlow Core는 완벽한 프로그래밍 제어 기능을 제공합니다.
TensorFlow Core는 기계 학습 연구자 및 모델을 정밀하게 제어해야하는 사람들에게 권장됩니다.
높은 수준의 API는 TensorFlow Core 위에 구축됩니다.
이러한 상위 수준의 API는 일반적으로 TensorFlow Core보다 배우고 사용하기가 쉽습니다.
또한 상위 수준의 API는 반복적인 작업을 여러 사용자간에 보다 쉽고 일관되게 만듭니다.
tf.contrib.learn과 같은 고급 API를 사용하면 데이터 세트, 견적 도구, 교육 및 추론을 관리 할 수 ​​있습니다.
상위 수준 TensorFlow API 중 일부 (메소드 이름이 포함 contrib)는 아직 개발 중입니다.
contrib이후의 TensorFlow 릴리스에서 일부 메서드가 변경되거나 더 이상 사용되지 않을 수도 있습니다.

# 텐서란?
TensorFlow에서 데이터의 중심 단위는 텐서 입니다.
텐서는 임의의 수의 차원으로 배열된 값들의 집합으로 구성됩니다.
텐서의 랭크는 차원의 개수입니다.
다음은 텐서(tensors)의 몇 가지 예입니다.
```
3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

# 텐서플로우 CORE

- 텐서플로우 가져오기

TensorFlow 프로그램에 대한 표준 import 문은 다음과 같습니다.
```
import tensorflow as tf
```
이렇게하면 파이썬은 TensorFlow의 모든 클래스, 메소드 및 심볼에 액세스 할 수 있습니다.
대부분의 문서에서는 이미 이 작업을 수행했다고 가정합니다.

- 연산 그래프

TensorFlow Core 프로그램은 두 개의 개별 섹션으로 구성되어 있다고 생각할 수 있습니다.
1. 연산 그래프 작성.
2. 연산 그래프를 실행합니다.
연산 그래프는 노드의 그래프로 배열 TensorFlow의 일련의 동작입니다.
간단한 전산 그래프를 작성해 봅시다.
각 노드는 0개 이상의 텐서를 입력으로 사용하고 출력으로도 생성합니다.
상수도 노드중 하나의 유형입니다.
TensorFlow 상수는 input과 output이 없으며 내부적으로 저장하는 값을 출력합니다.
node1과 node2라는 두개의 텐서를 만들어보겠습니다.
```
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
```
결과는 아래와 같습니다.
```
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```
우리의 예상과는 다른 결과가 나왔습니다.
3.0과 4.0이 나올줄 알았는데 이상한 상태를 가르키는듯한 문구만 나왔습니다.
우리가 원하는 값을 출력하려면 세션을 주고 실행을 시켜야 합니다.
아래의 내용을 추가해봅시다.
```
sess = tf.Session()
print(sess.run([node1, node2]))
```
Tensor노드를 연산과 결합 하여보다 복잡한 계산을 할 수 있습니다 (연산도 노드입니다).
예를 들어 두 개의 상수 노드를 추가하고 다음과 같이 새 그래프를 생성 할 수 있습니다.

```
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))
```
출력 결과값은 아래와 같습니다.

```
node3:  Tensor("Add_2:0", shape=(), dtype=float32)
sess.run(node3):  7.0
```
- 텐서보드

TensorFlow는 전산 그래프의 그림을 표시 할 수있는 TensorBoard라는 유틸리티를 제공합니다.
다음은 TensorBoard가 그래프를 시각화하는 방법을 보여주는 스크린 샷입니다.
![텐서플로우보드](https://deeptensorflow.github.io/images/tfboardex1.png)
데이터가 많아지면 일일히 정보를 확인하는 것이 어려울 수 있습니다.
텐서보드를 이용하면 원하는 데이터를 한눈에 볼 수 있습니다.

- placeholder

placeholder는 나중에 값을 채워주겠다고 하고 우선 형식만 선언해 두는 형태입니다.
아래와 같이 연산을 지정해 준 뒤에 feed_dict 매개 변수를 사용하여 이러한 입력란에 구체적인 값을 제공하는 Tensors를 지정하여 이 그래프를 여러 입력으로 평가할 수 있습니다.
```
import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

tf_add = a + b

sess = tf.Session()

print(sess.run(tf_add, feed_dict= {a : [1,2], b: [3,4]}))
```

- 학습 시키기

머신러닝 라이브러리인 만큼 당연히 모델을 만들고 학습을 시킬 수 있는 방법도 제공하고 있습니다.
모델을 학습 가능하게 만들려면 동일한 입력으로 새로운 출력을 얻기 위해 그래프를 수정할 수 있어야합니다.
변수를 사용하면 그래프에 학습 가능한 매개 변수를 추가 할 수 있습니다.
그것들은 타입과 초기 값으로 구성됩니다.

우선 처음에는 아래와 같이 평가 모델을 작성합니다.
```
import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([.3], tf.float32)

x = tf.placeholder(tf.float32)


Y = W * x + b

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(Y, feed_dict= {x : [1., 2., 3., 4.]}))
```
상수는 호출 할 때 초기화되며 tf.constant값은 절대로 변경 될 수 없습니다.
반대로 변수(tf.Variable)는 초기화 할 때 초기화되지 않습니다.
TensorFlow 프로그램의 모든 변수를 초기화하려면 다음과 같이 명시적으로 특수 작업(tf.global_variables_initializer())을 호출해야합니다.

우리는 모델을 만들었지만 아직 얼마나 좋은지 모릅니다.
교육 데이터에 대한 모델을 평가하려면 원하는 값을 제공하기 위한 목표값이 필요하며 손실 함수를 작성해야합니다.
손실 함수는 목표값으로부터 현재 모델이 얼마나 떨어져 있는지를 측정합니다.
현재 모델과 목표값 사이의 델타의 제곱을 합한 선형 회귀에 표준 손실 모델을 사용합니다.
즉, 기대값과 목표값의 차를 제곱한 값을 손실 모델로 작성합니다.
```
import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([.3], tf.float32)

x = tf.placeholder(tf.float32)
_y = tf.placeholder(tf.float32)

Y = W * x + b
loss = tf.reduce_sum(tf.square(Y-_y))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(loss, feed_dict= {x : [1, 2, 3, 4], _y : [0, -1, -2, -3]}))
```

우리는 tf.assign을 통해 tf.Variable로 선언된 변수에 대해서 수동으로 값을 바꾸어 줄 수 있습니다.
예를들어 위의 모델은 W가 -1, b가 1 일때 손실 함수가 0이 되어 최적의 매개변수가 됩니다.
아래와 같이 모델을 작성할 수 있습니다.
```
import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([.3], tf.float32)

x = tf.placeholder(tf.float32)
_y = tf.placeholder(tf.float32)

Y = W * x + b
loss = tf.reduce_sum(tf.square(Y-_y))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fixW = tf.assign(W, [-1])
fixb = tf.assign(b, [1])

sess.run([fixW, fixb])

print(sess.run(loss, feed_dict= {x : [1, 2, 3, 4], _y : [0, -1, -2, -3]}))
```
print값은 0이 됩니다.
우리는 Wand 의 "완벽한"값을 추측 b했지만 기계 학습의 요점은 올바른 모델 매개 변수를 자동으로 찾는 것입니다.
다음 섹션에서 이를 수행하는 방법을 보여줄 것입니다.

- 트레이닝 시키기

텐서플로우에서는 손실함수의 값을 최소화 시키기 위해서 여러 optimizer들을 제공합니다.
이들을 간단한 API로 손실함수를 최소화 시킵니다.
가장 간단한 예는 gradient descent optimizer입니다.
```
import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([.3], tf.float32)

x = tf.placeholder(tf.float32)
_y = tf.placeholder(tf.float32)

Y = W * x + b
loss = tf.reduce_sum(tf.square(Y-_y))

init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, feed_dict={x : [1, 2, 3, 4], _y : [0, -1, -2, -3]})

print(sess.run([W, b]))
```
위와 같은 방법으로 자동으로 W, b를 학습시킬 수 있습니다.

# tf.contrib.learn
tf.contrib.learn는 기계 학습의 메커니즘을 단순화하는 고급 TensorFlow 라이브러리입니다.
평가관련 반복문 관리, 트레이닝 관련 반복문 관리, 데이터셋 관리, feeding관리를 포함합니다.
tf.contrib.learn은 많은 공통 모델을 정의합니다.

linear regression코드에 이를 적용하면 아래와 같이 간결해집니다.
```
import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
estimator.evaluate(input_fn=input_fn)
```

# 커스텀모델 만들기
tf.contrib.learn는 함수정의부분을 수정할 수 있습니다.
TensorFlow에 내장되어 있지 않은 커스텀 모델을 만들고 싶다고 가정 해 보겠습니다.
우리는 여전히 높은 수준의 반복문 관리, 트레이닝 관련 반복문 관리, 데이터셋 관리, feeding관리를 유지할 수 있습니다.
설명을 위해, 우리는 저수준 TensorFlow API에 대한 지식을 사용하여 LinearRegressor에 대한 자체 모델을 구현하는 방법을 보여줄 것입니다.
tf.contrib.learn.Estimator를 써서 개발해 보도록 하겠습니다.
tf.contrib.learn.LinearRegressor는 tf.contrib.learn.Estimator의 sub-class입니다.
Estimator에게 예측, 교육 단계 및 손실을 평가할 수있는 방법을 tf.contrib.learn에 알리는 function_fn이라는 기능을 제공하기 만하면됩니다.
아래의 코드가 그 예제입니다.

```
import numpy as np
import tensorflow as tf
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss= loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
x=np.array([1., 2., 3., 4.])
y=np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))
```
