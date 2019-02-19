# 텍스트와 시퀀스를 위한 딥러닝
- 이번 장에서는 텍스트, 시계열 등과 같은 Sequential한 데이터를 처리할 수 있는 딥러닝 모델을 살펴본다.
- 기본적으로 RNN(Recurrent Neural Network)와 1D Convent이 있다.

* 자세한건 코드 참고

| 핵심 키워드 |
| ---------- |
| Sequential Data |
| One-Hot Encoding |
| Embedding |
| RNN |
| LSTM |
| Generator function |
| GRU |
| Dropout |

### 자연어 처리..?
- 문자 언어에 대한 통계적 구조를 만들어 간단한 텍스트 문제를 해결한다.
- 앞선 5장의 컴퓨터 비전이 픽셀에 적용한 pattern recognition이라면, 
- 6장은 단어, 문장, 문단에 적용한 pattern recognition이다.
- 텍스트 자체를 입력으로 넣을 수 없기 때문에, 이를 수치화 해주는 작업을 해줘야한다.(vectorizing)
- n-gram 추출하여 vector로 만들어주는데, 예를 들면
- 'the cat sat on the tree'라는 text가 있다면, {"the", "the cat", "cat", "cat sat", 'the cat sat",......,"on the tree"}로 BoW를 만들 수 있다.
- 하지만, 이번장에선 n-gram에 대해 제대로 다루진 않는다.

### 텍스트를 수치화하는 과정 (Text --> Token --> Vector)
  ##### Tokenize : Text --> Token
  - 텍스트를 나누는 단위를 token이라 한다. (단어(word) / 문자(character) / n-gram)
  ##### Vectorize : Token --> Vector
  - token과 vector를 연결한다.
  - 수치가 아닌 것을 수치로 만든다.
  - 크게 one-hot encoding과 embedding이 있다.
  

### One-Hot Encoding
- 벡터의 i번째 원소만 1이고 나머지는 모두 0인 형태
- 딱봐도 굉장히 sparse한 것을 알 수 있다.
- token 수가 많아지면 자연스레 벡터의 차원도 증가하게 된다.
- 고유 token의 수가 너무 많아 다루기 힘들다면 one-hot hashing을 사용할 수 있다. 
- 인덱스를 딕셔너리에 저장하는 대신 단어를 hashing하여 랜덤 인덱스로 변환한다.

### Embedding : 단어간의 의미관계를 기하학적 변환으로 인코딩
- one-hot encoding은 딕셔너리에 있는 단어의 수가 곧 차원의 수다.(굉장히 sparse)
- 이에 비해 word embedding은 dense word vector를 사용하여 비교적 저차원.
- word embedding은 얻은 데이터로부터 학습된다.
- embedding은 직접 layer를 학습시킬 수 있고, pre-trained word embedding을 사용할 수 있다.
- 단어를 dense vector로 연관짓는 간단한 방법은 random vector를 선택하는 것이지만, 이는 embedding 공간이 구조적이지 않다.
- 구조적이지 않다는 것은 예를들어 비슷한 의미를 가진 'accurate'와 'exact'가 완전히 다른 embedding을 가진다는 의미다.
- **단어사이의 의미 관계를 반영해야한다. 비슷한 단어는 가까이에 embedding되도록...**
- 법률문서 분류를 위한 embedding과 영화리뷰 분류를 위한 embedding은 다르다. 의미관계의 중요성이 목적마다 다르다.

### Embedding 구조
- 단어 인덱스(2D Tensor) --> Embedding Layer --> 연관된 word vector(3D Tensor)
- (samples, word_index의 나열) shape으로 들어와 (samples, word_index의 나열, embedding dimensionality)로 나간다.
- 즉, 각 단어별로 정해준 차원의 vector로 표현되는 것이다.
- Embedding Layer의 가중치는 (빈도수높은 단어 상위 몇개, embedding 차원)이고 biases는 없다.
- 그래서 Embedding의 인자는 input_dim, output_dim(:embedding dim), maxlen이 대표적이다. ;maxlen은 사용할 텍스트의 길이, 이후로는 자른다.
- preprocessing.sequence.pad_sequences()가 주로 전처리에서 사용된다. 
- **pre-trained word embedding**은 load하여 `model.layers[0].set_weights([embedding_matrix])`로 넣어주고
- embedding layer는 동결(freezing)시켜준다. 학습되지 않도록...

### IMDB Modeling (1) : 데이터 200개
  ##### 1. embedding layer + 2 Dense layers using GloVe
  - train sample을 200개로 한정하여 진행하였다.
  - GloVe를 embedding layer의 weight로 setting하였다.
  - validation accuracy(50% 후반대)가 잘 나오지 않았고, overfitting이 빠르게 시작.

  ##### 2. embedding layer + 2 Dense layers without the pre-trained 
  - train sample은 그대로 200개.
  - validation accuracy(50% 초반대)는 역시 잘나오지 않았고, GloVe를 쓴게 좀더 성능이 좋았다.
  
  ##### 3. sample 2000개로 진행
  - 책에는 70%의 validation성능이 나올거라했는데....나아진게 없었다.
  
### Recurrent Neural Network (RNN)
- 이전까지 사용했던 Fully Connected Network(FCN), CNN은 Feedforward Network에 해당한다.
  - sequence가 한번에 처리된다.
  - 입력간에 유지되는 상태가 없이, 입력이 개별적으로 처리됨(memory개념이 없다).
- RNN은 sequence를 하나의 데이터 포인트가 되는건 feedforward network와 동일하지만,
- RNN network가 sequence의 원소를 차례대로 방문한다.(sequence가 한번에 처리되지 않음)
- (timesteps, input_features)인 2D tensor를 입력으로 받는다. --> (batch_size, timesteps, input_features)로 들어가겠지?
- timestep마다 동일한 weight가 적용된다.(timestep마다 동일한 RNN레이어를 들어가니 어찌보면 당연한 얘기)

### IMDB Modeling (2) : 전체 데이터
  ##### 4. Embedding layer + SimpleRNN layer + dense layer(출력)
  - maxlen을 좀더 늘려 500으로 설정하고 데이터를 재정제하였다.
  - 전체데이터를 사용한 결과 validation 성능은 max 85%가 나왔다.
  - 이는 이전 3장에서 FCN으로 classification한 성능 87%에 못미치는 결과다.
  - 이유는 FCN보다 얻은 정보가 부족(처음 500개 단어이후로 자름), 긴 텍스트를 처리하기에 simpleRNN은 적합하지 않다.

### Long Short Term Memory (LSTM)
- simpleRNN은 긴 timesteps에 걸친 의존성은 학습할 수가 없다. 왜?
- timestep마다 역전파시 동일한 가중치가 곱해지면서 gradient가 vanishing or exploding하게 된다.(feedforward network와 동일한 문제)
- 여기엔 cell state의 개념이 도입된다.
  - 정보를 여러 timesteps에 걸쳐 나르는 방식인데, 오래된 시그널이 점차 소멸되는 것을 막아준다.
- 3개의 gate와 2개의 state가 존재하는데, 이는 [여기](https://imjuno.tistory.com/entry/RNN)를 참고.(책으로 파악하기 쉽지않음)
- parameter의 갯수는 어떻게 산정될까? --> [(units 수 + input_dim + 1(for bias) x units 수] x 4 (GRU의 경우 3)

### IMDB Modeling (3) : 전체 데이터
  ##### 5. Embedding layer + LSTM layer + dense layer(출력)
  - 88%~89%정도의 validation 성능이 나왔다. (Gradient vanishing에 덜 영향받기 때문)
  - 계산이 많은 점을 고려하면 월등히 높진 않음.
  - 이유는 hyperparameter tuning 미실시, regularization이 없었기 때문.
  - 그리고 리뷰를 길게 분석하는 것은 classification문제에서는 큰 도움이 되지 않는다. (FCN이 더 효율적!)
  - RNN은 Question-Answering이나 Machine-Translation쪽에 더 적용하기 좋다.
  
## Advanced RNN (고급 사용법)
  ### Recurrent Dropout
  - RNN layer의 overfitting을 막기위해 dropout 사용 (layer가 아닌 매개변수setting값으로 설정)
  - timestep마다 동일한 unit을 drop해줘야한다. (이렇게 안하면 weight학습이 안되겠지?)
  - dropout과 recurrent_dropout이 있다. (각각 input에 대한 dropout 비율 / (hidden)state에 대한 dropout 비율)
