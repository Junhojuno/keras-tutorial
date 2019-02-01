## 5장 정리
- 주된 내용은 Convolutional Neural Network (CNN) ; 합성곱
- 기본적인 stride, pooling 등의 개념은 안다는 가정하에 들어감.
- 새롭게 이해한 내용위주로 정리 ; 자세한건 책을 다시 읽어볼 것!

| 핵심 키워드 | 설명 |
| ------------- |:-------------:|
| 합성곱 | 2D filter를 통해 진행되는 연산 |
| 2D filter/window | 특정 패턴; 합성곱에 사용된다. |
| feature map | 합성곱 연산으로 나오는 3D 텐서; 이미지에서 (높이,너비,채널 수(깊이))로 표기된다. |
| Pooling/sub-sampling | 특성 맵의 크기를 줄이는 과정 |
| Data Augmentation | 학습 데이터가 적을 때, overfitting을 막기위해 데이터를 늘리는 기법 |
| Pre-Trained Network | 이미 학습된 network의 특성을 가져와 사용하였다. ; 여기선 VGG16 |
| Feature Extraction |  |
| Fine-Tuning |  |

### Fully Connected Layer (FCN) vs Convolutional Neural Network (CNN)
- 이전 FCN으로 MNIST 분류를 했을 때, 테스트 정확도는 97%정도였다.(CNN은 99%)
- 이러한 차이를 보이는 이유는 패턴학습방식에 있다.
- FCN은 픽셀 하나하나의 패턴(전역패턴)을 학습하지만, CNN은 2D filter(window)로 일부 패턴(자역패턴)만 학습한다.

### 일부만 학습한게 왜 더 좋은 걸까?
- 일부만 학습한다는건 다음과 같다.
- CNN이 이미지의 오른쪽 윗모서리에서 어떤 패턴을 학습했다면, 다른 곳에서도 이 패턴을 인식할 수 있다.
- FCN의 경우, 위치가 바뀌면 학습도 새로해야한다.
- 적은 수의 샘플로 일반화하는 능력을 학습할 수 있다는 것이다.
- 또 하나, 합성곱 layer가 늘어날수록 복잡하고 추상적인 이미지를 효과적으로 학습할 수 있다.
- 다시 말해, 첫번째 합성곱 layer에서 edge나 texture(질감) 등의 지역패턴을 학습했다면,
- 두번째 합성곱 layer에서는 첫번째 layer보다는 큰 개념의 패턴을 학습하게 된다.
- 이를 'CNN은 패턴의 공간적 계층구조를 학습할 수 있다.'고 한다.

### Convolutional Neural Network (CNN)
- filter라고 하는 2D 텐서와 이미지를 연산하여 지역적인 패턴을 학습,
- 깊게 layer를 들어갈수록 지역 패턴(턱의 라인, 귀 라인) --> 추상적인 개념(눈, 귀)을 학습한다.
- convolution layer의 첫번째 매개변수(parameter)는 filter 갯수를 의미하고,
- 첫 입력 이미지의 채널수는 합성곱을 거치게 되면, 더이상 RGB의 특정 컬러를 의미하지 않게 된다. 
  - 단지, 합성곱 연산을 통해 만들어진 feature map의 갯수가 된다.(2D인 feature map이 쌓여서 3D 텐서 형태가 된다고 보면 될 거 같다.)
  - 코드로 보면, Conv2D의 첫 매개변수인 output_depth가 이걸 의미한다.
  - 예를 들면, 4x4 컬러 이미지, 3x3 filter, stride=1, padding=1인 경우
  - ex) (batch_size, 4, 4, 3) x (3,3,3,32) --> (batch_size, 4, 4, 32) ; 이미지의 4번째 3과 filter의 3번째 3을 맞춰줘야한다.
- feature map의 깊이는 점점 증가하지만, 크기는 계속 감소하게 되는게 전형적인 convolution network의 패턴.

### 왜 pooling으로 feature map의 크기를 줄이는 거지?
- 합성곱을 진행하는 중간중간 max pooling이라는 층이 들어간다.
- 이는 층마다 feature map의 크기를 절반(보통 절반)으로 줄이는 sub-sampling과정이다.
- 이걸하는 이유는 크게 2가지라고 한다.
- 먼저, 그냥 합성곱만하면 공간적 계층 구조학습에 도움이 안된다고 한다.
  - 그냥 합성곱만하면 초기 입력값에서 얻은 지역패턴 정보만 계속 전달되어, 추상적 개념 학습이 이루어지지 않는다고 한다.
  - 지역패턴을 넘어 더 큰 패턴의 정보를 학습하기 위해선 sub-sampling이 필요하다는 것!
  - 연속적인 합성곱 layer가 상대적으로 커지는 2D filter를 통해 학습하도록 구조를 구성해야한다. 
- 둘째는 parameter수가 많아져 학습에 효율성이 떨어지고, overfitting도 심하게 일어난다고 한다.

### 훈련 샘플이 적은 경우엔 overfitting이 중요한 문제다! --> Data Augmentation으로 해결해보자!
- 5장 p192처럼 적은 데이터를 그냥 학습했을땐, 이진분류에서 validation acc가 70%정도 (overfitting은 기본^^)
- 이번 장에서는 Data Augmentation으로 이를 해결해나갔다. (이전 장에서는 dropout, L2 regularization 등 사용)
  - 기존 훈련 샘플을 가지고 더 많은 훈련 데이터를 생성해내는 방식
  - ImageDataGenerator() 사용
  - 같은 데이터가 두번 입력되지 않으면서 많은 데이터를 학습시킬 수 있다.
  - 하지만, 데이터간에 상호 연관성이 있기 때문에 overfitting이 충분히 해소되지 않았을 것이다.
  - 그래서 p196에서 FCN쪽에 Flatten다음에 dropout을 추가해주었다.
  
### 훈련 샘플이 적은 경우에 성능을 어떻게 끌어올릴 것인가?
- pre-trained network를 사용한다.; 학습된 특성을 다른 문제에도 적용할 수 있다.
- 사용방식은 크게 2가지로 나뉜다.
- 먼저, feature extraction ; pre-trained network의 합성곱 layer를 사용 (training X)
  - 다시말해, pre-trained network의 FCN부분만 내가 학습하고 합성곱 부분은 빌려온다.
  - 이렇게 쓰는 이유는 분류기(FCN부분)은 해당 모델의 클래스에 특화되어있기 때문이다.
  - 분류기(FCN부분)는 전체 사진에 대해 어떤 클래스에 속할 확률 정보만 담겨 있다.
  - 또한, 이 부분에는 더이상 입력이미지의 객체(눈, 귀 등등)의 위치정보가 담겨있지 않다.
  - 객체의 위치가 중요한 경우, 굳이 FCN부분에서 만든 feature는 크게 쓸모가 없다.
  - 이 방식도 두가지로 운영할 수 있다.
    - 첫번째, 입력이미지를 pre-trained network 합성곱 부분에 넣어 최종feature를 뽑고, FCN 학습
    - 합성곱 layers를 한번만 통과하면 됨. **but 이건 data augmentation을 사용하지 못한다!!!**
    - 다음은, pre-trained 합성곱 layer + FCN을 연결해 end-to-end로 training시킨다.
    - 이 과정은 pre-trained 합성곱 layer의 feature가 바뀌지 않도록 동결(freezing)시켜줘야함.
 - 다음은, fine-tuning ; pre-trained network의 일부 합성곱 layer(상위 2~3개)를 training에 포함
  - pre-trained network의 합성곱 layer 상위층 일부를 FCN과 함께 재훈련시키는 것이다.
  - 왜 상위층 2~3개만 fine-tuning하는 것인가?
  - 하위층일수록 좀 더 일반적인/재사용가능한 특성들을 인코딩하지만, 상위층일수록 좀 더 해당문제에 특화된 특성을 인코딩한다.
  - 새로운 문제에 적용하기위해서 상위층을 조정하는 것이 바람직하다. (하위층으로 갈수록 fine-tuning의 효과는 떨어짐.)
  - 또한, 하위층으로 갈수록 훈련해야할 parameter가 많아져 overfitting의 위험이 있다.
  
