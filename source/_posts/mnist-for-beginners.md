---
title: 초보자를 위한 MNIST(텐서플로우)
categories:
  - Tensorflow
  - App
tags:
  - tensorflow
  - MNIST
date: 2017-02-28 09:17:40
thumbnail: https://deeptensorflow.github.io/images/logo.png
---
이 글은 https://www.tensorflow.org/get_started/mnist/beginners 공식 홈페이지 내용을 번역한 것이 주 내용입니다.

# MNIST란 무엇일까?
MNIST란 본래 미국에서 우편번호를 분류할때 기계에게 분류를 맡기려는 시도에서 비롯되었습니다.
MNIST는 간단한 컴퓨터 시각 데이터 세트입니다.
이것은 다음과 같은 자필 자릿수의 이미지로 구성됩니다.
![MNIST](https://deeptensorflow.github.io/images/mnist.png)
또한 각 이미지의 레이블을 포함하여 어떤 숫자인지 알려줍니다.
예를 들어 위 이미지의 레이블은 5, 0, 4 및 1입니다.
이 자습서에서는 이미지를보고 어떤 자릿수인지 예측하는 모델을 교육 할 것입니다.
우리의 목표는 최첨단 성능을 구현하는 매우 정교한 모델을 교육하는 것이 아닙니다.
우리는 Softmax Regression이라 불리는 매우 간단한 모델로 시작할 것입니다.
이 자습서의 실제 코드는 매우 짧으며 모든 흥미로운 내용은 단 3 줄에서 발생합니다.
그러나 TensorFlow가 작동하는 방법과 핵심 기계 학습 개념 모두에 대한 아이디어를 이해하는 것이 매우 중요합니다.

이 튜토리얼에서 우리가 달성 할 수있는 것 :

- MNIST 데이터와 softmax 회귀에 대해 알게 될 것입니다.

- 이미지의 모든 픽셀을 보면서 숫자를 인식하는 모델 인 함수를 만들 것입니다.

- TensorFlow를 사용하여 수천 가지 예제를 "훑어보고"숫자를 인식하도록 모델을 교육합니다(첫 번째 TensorFlow 세션을 실행하여 수행).

- 테스트 데이터로 모델의 정확성을 확인할 것입니다.

MNIST 데이터는 Yann LeCun의 웹 사이트 에서 호스팅됩니다.
이 자습서의 코드를 복사하여 붙여 넣는 경우 다음 두 줄의 코드를 사용하여 데이터를 자동으로 다운로드하고 읽습니다.

```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```
MNIST 데이터는 훈련 데이터(mnist.train) 55,000개, 테스트 데이터(mnist.test) 10,000개, 및 검증 데이터(mnist.validation) 5,000개 세 부분으로 나누어집니다.
이 분할은 매우 중요하며 기계 학습에서 필수적입니다.
이미 트레이닝을 한 데이터에 대해서 검증을 하고 테스트를 하는 것은 의미가 없겠죠?
꼭 나누어주세요!
MNIST데이터는 두가지로 분류되어 있습니다.
손으로 직접 쓴 글씨와 그에 해당하는 라벨 데이터로 나누어져 있습니다.
트레이닝 이미지는 mnist.train.images이고 트레이닝 레이블은 mnist.train.labels입니다.
트레이닝 세트와 테스트 세트에는 이미지와 해당 레이블이 포함되어 있습니다.
![mnistmatrix](https://deeptensorflow.github.io/images/mnistmatrix.png)
이 배열을 28x28 = 784 숫자의 벡터로 전개 할 수 있습니다.
이 배열을 1열로 펼친다면 784개의 숫자로 이루어진 데이터가 되겠죠?
55000개의 데이터를 테스트 데이터로 쓸 때 [55000,784] 배열꼴의 데이터를 쓴다면 MNIST데이터 55000의 데이터를 모을 수 있는 꼴이 될 것입니다.
첫 번째 차원은 이미지 목록에 대한 인덱스이고 두 번째 차원은 각 이미지의 각 픽셀에 대한 인덱스입니다.
![mnist데이터셋행렬](https://deeptensorflow.github.io/images/mnisttrainxs.png)
MNIST의 각 이미지에는 이미지에 그려지는 숫자를 나타내는 0에서 9 사이의 숫자가 해당 레이블을 가지고 있습니다.
이 튜토리얼의 목적을 위해 우리는 레이블을 "원 핫 벡터 (one-hot vectors)"로 할 것입니다.
원 핫 벡터는 대부분 차원에서 0이고 단일 차원에서 1 인 벡터입니다.
이 경우,n번째 자릿수는 n차원 벡터로 표현됩니다.
예를 들어, 3은 [0,0,0,1,0,0,0,0,0,0] 입니다.
자, 이제 softmax를 통해서 분석할 일만 남았습니다.

# softmax란?
 우리는 이미지를보고 각 숫자가 될 확률을 줄 수 있기를 원합니다.
 예를 들어, 우리 모델은 9의 그림을보고 80%의 확률로 9라고 확신 할 수 있지만, 다른 모든 것들에도 약간의 확률이 있습니다(합계 20%).
 100% 확실하지는 않습니다.
 이렇게 각 숫자에 대한 확률로 나타내주는 쉬운 방법이 softmax 회귀분석을 쓰는 방법입니다.
 Softmax 회귀 분석을 하는 과정을 아래에서 보여드리겠습니다.
 우선 [55000, 784] 행렬의 데이터가 들어올 것입니다(55000개의 MNIST데이터셋).
 그리고 이 데이터셋은 [784,10]꼴의 행렬과 곱해지게 될 것입니다(weight의 개념으로).
 그리고는 [10]꼴의 행렬과 더해질 것입니다(bias의 개념으로).
 그러면 각 데이터셋에 대해서 [10]꼴의 배열에 임의의 숫자가 나오겠지요?
 그것을 softmax로 처리하면 각각 10개의 데이터가 0~1사이의 데이터로 변합니다.
 그 10개의 데이터값을 다 합한뒤 각각의 데이터에 나누어준다면 그것이 확률이 되는 것입니다.
 ![softmax](https://deeptensorflow.github.io/images/tf_softmax.png)

# 텐서플로우를 이용해 MNIST를 분석해보자!
이전의 포스팅에서 설명드렸던 placeholder를 이용해 값을 유동적으로 받기 위한 x를 선언하겠습니다.
```
x = tf.placeholder(tf.float32, [None, 784])
```
위의 코드에서 None에는 데이터의 개수가 올 것입니다.

그리고 위에서 설명한대로 softmax를 정의하기 위해서 [784,10] 꼴의 weight와 [10]꼴의 bias가 필요합니다.
```
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```
그럼 이제 x값에 weight와 bias를 더하고 softmax를 취해줘야겠죠?
텐서플로우에서는 이것이 한줄로 가능합니다.

자, 이제 데이터를 받고 softmax를 적용시킬 준비가 끝났습니다.

# 트레이닝 시키기
우리는 우선 트레이닝을 시키기 위해서 손실함수를 작성해야합니다.
이는 모델이 원하는 결과와 얼마나 멀리 떨어져 있는지 나타냅니다.
우리는 손실함수를 최소화하려고 노력하며, 손실함수 마진이 작을수록 모델이 더 좋습니다.
어렵게 생각하지 마세요!
손실함수는 그냥 실제 값과 우리의 예측값이 얼마나 다른지 알 수 있게 해주는 함수입니다.
여기서는 인기있는 손실함수인 cross-entropy라는 것을 써보겠습니다.
자세한 것은 모르셔도 되고 그냥 실제 값과 예측 값의 차이를 표현하는 한 방식이라고 보시면 됩니다.
y 값은 예측 값입니다.
y_ 값이 실제 값입니다.
```
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
```
그런 다음 이 손실함수를 통해서 예측값과 실제값의 차이를 알았으니 차이를 좁혀주기 위해서 학습을 시켜야겠지요?
GradientDescentOptimizer 라는 기본적인 알고리즘으로 학습을 시켜보겠습니다.
마찬가지로 이 알고리즘이 어떤식으로 동작하는지 모르더라도 텐서플로우에서 제공하는 API를 쓰면 손쉽게 적용할 수 있습니다.
```
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```
이제 반복문으로 이 트레인함수를 돌려주면 됩니다.
mnist.train.next_batch는 텐서플로우에서 제공해주는 기본 API입니다.
데이터를 100개씩 랜덤으로 뽑아줍니다.
그리고 placeholder로 x와 y_를 선언했기 때문에 데이터를 읽어서 바로 x와 y_에 넣어줄 수 있습니다.
그런뒤 위의 손실함수를 GradientDescentOptimizer로 최소화시켜주는 것입니다(train).
```
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
```

# 모델 평가하기
자, 이렇게 훈련시킨 모델이 얼마나 잘 학습되었는지 확인해봐야겠죠?
softmax를 거친 우리 기대값 배열은 0~9사이의 숫자가 나올 확률을 계산해 두었을 것입니다.
그렇다면 확률이 가장 높게 나온 label이 우리의 예측 숫자가 될 것입니다.
테스트 데이터를 넣고 정확도를 평균내어봅시다.
```
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

정확도가 몇퍼센트가 나오시나요?
제 코드를 기준으로는 91.4프로가 나옵니다.
다음에는 이 정확도를 99프로 이상으로 끌어올려보도록 하겠습니다.
