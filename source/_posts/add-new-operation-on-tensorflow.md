---
title: 텐서플로우에 새로운 연산 추가하기(커스터마이징)
categories:
  - Tensorflow
  - Internal
tags:
  - tensorflow
  - internal
  - build
date: 2017-02-21 22:24:53
thumbnail: https://deeptensorflow.github.io/images/logo.png
---
텐서플로우를 이용해서 소스코드를 작성하다보면 가끔 내가 만든 연산을 추가하고 싶다는 생각이 듭니다.
이럴 경우에 텐서플로우에서는 operation을 추가하는 방법을 공식홈페이지에서 설명해주고 있습니다.
https://www.tensorflow.org/extend/adding_an_op
하지만 영어로 되어있기에... 개인 저장용으로 operation을 텐서플로우에 추가하는 방법을 적어보고자 합니다.

# operation 추가하기
기존 TensorFlow 라이브러리에서 다루지 않는 연산을 만들려면 먼저 파이썬에서 연산을 기존 파이썬 연산이나 함수의 조합으로 작성하는 것이 좋습니다.
이것이 가능하지 않으면 사용자 정의 C ++ op를 작성할 수 있습니다.
사용자 정의 C ++ 연산을 작성하는 데에는 여러 가지 이유가 있습니다.

- 기존 작업의 구성으로 작업을 표현하는 것이 쉽지 않거나 가능하지 않은 경우
- 기존 기본 요소의 구성으로 작업을 표현하는 것은 효율적이지 않은 경우
- 원래 있던 기존의 요소들을 미래의 컴파일러가 융합하기 힘들어하는 경우
즉, 왠만하면 기존의 텐서플로우 op로 연산을 하되 기존의 op로 연산이 불가능하거나 효율적이지 않은 경우 본인의 op를 직접 추가하라는 말입니다.

맞춤 작업을 통합하려면 다음 작업이 필요합니다.

1. 새로운 op를 C ++ 파일로 등록하십시오. Op 등록은 op의 구현과 독립적인 op 기능을위한 인터페이스를 정의합니다. 예를 들어 op 등록은 op의 이름과 op의 입력과 출력을 정의합니다. 또한 텐서의 모양을 정의합니다.
2. C ++로 op의 실제 동작을 구현하십시오. op의 구현은 커널로 알려져 있으며 1 단계에서 등록한 인터페이스의 구체적인 구현입니다.(실제 연산을 정의하라는 말) 다양한 입/출력 유형 또는 아키텍처 (예 : CPU, GPU)에 대해 여러 개의 커널이있을 수 있습니다.
3. Python 래퍼를 만듭니다 (선택 사항). 이 래퍼는 Python에서 op를 만드는 데 사용되는 공용 API입니다. op 등록에서 기본 래퍼가 생성됩니다.이 래퍼는 직접 사용하거나 추가 할 수 있습니다.
4. op (옵션)의 gradient를 계산하는 함수를 작성합니다.
5. op를 테스트하십시오. 우리는 대개 편의상 Python에서 이 작업을 수행하지만 C++로 op를 테스트 할 수도 있습니다. Gradient를 정의하면 파이썬 gradient checker로 확인할 수 있습니다. relu_op_test.pyRelu를 보면 Relu-like operators의 forward함수와 그들의 gradient를 확인할 수 있습니다.

# op의 인터페이스 정의
op의 인터페이스는 TensorFlow 시스템에 등록하여 정의합니다.
등록시 op의 이름, 입력 (유형 및 이름) 및 출력 (유형 및 이름)과 op가 필요할 수 있는 docstrings 및 attrs를 지정합니다.
이것이 어떻게 작동 하는지를보기 위해 예시를 들어보겠습니다.
int32 형태의 첫 번째 요소를 제외한 모든 요소가 0으로 되는 텐서 복사본을 출력 하는 op를 만들고 싶다고 가정합니다.
이렇게하려면 명명 된 파일을 만듭니다 zero_out.cc.
그런 다음 REGISTER_OP사용자 인터페이스에 대한 인터페이스를 정의하는 매크로 호출을 추가 하십시오.
~tensorflow/core/user_ops 에 파일을 만들었습니다.
```
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```
이 ZeroOut연산은 하나의 텐서 to_zero32 비트 정수를 입력으로 취해서 텐서 32 비트 정수 zeroed을 출력합니다.
op는 출력 텐서가 입력 텐서와 동일한 모양인지 확인하기 위해 shape 함수를 사용합니다.
예를 들어, 입력이 텐서 형태 [10, 20]이면이 모양 함수는 출력 모양도 [10, 20]으로 지정합니다.

- 이름 지정에 대한 참고 사항 : op 이름은 CamelCase여야하며 binary file에 등록된 다른 모든 운영 체제 중에서 고유해야합니다.

# op의 kernel코드 작성
인터페이스를 정의한 후에 op의 하나 이상의 구현을 제공하십시오.
커널을 만드려면 OpKernel을 확장하는 클래스를 만들어야합니다.
그리고 Compute method를 오버라이드 해야합니다.
Compute메서드는 OpKernelContext* type의 context 인수를 하나 제공합니다.
이 인수를 사용하여 입력 및 출력 텐서와 같은 유용한 항목에 액세스 할 수 있습니다.
위에 작성한 파일에 커널을 추가하십시오. 커널은 다음과 같이 생겼습니다.
```
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = input(0);
  }
};
```
커널을 구현 한 후에는 TensorFlow 시스템에 등록하십시오.
등록시 이 커널이 실행될 다른 제약 조건을 지정합니다.
예를 들어, CPU 용으로 만든 커널 하나와 GPU 용으로 만든 커널을 따로 가질 수 있습니다.

그리고 이것을 ZeroOut op가 하기 위해서 아래의 코드를 zero_out.cc 코드에 추가합니다.
```
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

# op라이브러리 빌드
두가지 방법을 제시하고 있습니다.
g++을 통한 방법과 bazel을 통한 방법인데요.
bazel을 통한 빌드가 더 빠르다는 커뮤니티원의 정보를 듣고 bazel만으로 빌드를 진행했습니다.

TensorFlow 소스가 설치되어있는 경우 TensorFlow의 빌드 시스템을 사용하여 작업을 컴파일 할 수 있습니다.
디렉토리에 다음 Bazel 빌드 규칙이있는 BUILD 파일을 tensorflow/core/user_ops놓습니다.
```
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```
zero_out.so를 빌드하기 위해서는 아래의 명령어를 입력해주세요.
```
bazel build --config opt //tensorflow/core/user_ops:zero_out.so
```
참고 : .so표준 cc_library규칙 을 사용하여 공유 라이브러리(파일)를 만들 수 있지만 tf_custom_op_library매크로를 사용하는 것이 좋습니다. 몇 가지 필수 종속성을 추가하고 공유 라이브러리가 TensorFlow의 플러그인로드 메커니즘과 호환되는지 확인하기위한 검사를 수행합니다.

# 파이썬에서 op를 사용하기 위한 방법
TensorFlow Python API는 tf.load_op_library동적 라이브러리를로드하고 op를 TensorFlow 프레임 워크에 등록하는 기능을 제공합니다.
load_op_libraryop와 커널을 위한 파이썬 래퍼를 포함하는 파이썬 모듈을 리턴합니다.
따라서 일단 op를 빌드하면 다음을 수행하여 Python에서 실행할 수 있습니다.
```
import tensorflow as tf
zero_out_module = tf.load_op_library('zero_out.so')
with tf.Session(''):
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

# Prints
array([[1, 0], [0, 0]], dtype=int32)
```
참고로 snake_name case가 되기 때문에 op의 이름이 C++에서 ZeroOut이었다면, 파이썬에서는 zero_out이 됩니다.

# 테스트
성공적으로 op를 구현했는지 확인하는 좋은 방법은 테스트를 작성하는 것입니다.
다음 내용으로 zero_out_op_test.py 파일 을 만듭니다 .
```
import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.test.main()
```
그런 다음 테스트를 실행하십시오 (tensorflow가 설치되어 있다고 가정).
```
python zero_out_op_test.py
```

# 그런데 이대로 되시나요??
지금 op customizing 부분에 심각한 오류가 있는 것 같습니다.
bazel을 이용해서 op customizing을 하는 부분은 현재 오류가 있어서 안되는 것 같더군요.
그래서 g++로 다시 빌드를 해보았습니다.
```
# 홈디렉토리에서
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# tensorflow/tensorflow/core/uer_ops 경로에서
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -O2

```
자, 이렇게 빌드를 했더니 zero_out.so 파일이 해당 경로에 생겼습니다.
다시 파일로 돌아갔습니다.
그리고 path를 잘 못알아들어서 수동으로 다시 코드를 짜주었습니다.
```
import os.path
import tensorflow as tf
zero_out_module = tf.load_op_library('zero_out.so가 있는 경로 전부 루트부터 입력')
sess = tf.Session()
print (sess.run(zero_out_module.zero_out([[1,2], [3,4]])))
```
자, 이제 정상 작동을 합니다.

이것으로 user customizing op를 텐서플로우에 추가하는 방법에 대한 설명을 마치겠습니다.
