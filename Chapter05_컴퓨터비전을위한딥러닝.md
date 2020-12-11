## Chapter_05 컴퓨터 비전을 위한 딥러닝

### 5.1 합성곱 신경망 소개
- 컨브넷(convnet)이라고 불리우는 합성곱 신경망(convolutional neural network)은  
  컴퓨터 비전에 사용됨
- 컨브넷은 (image_height, image_width, image_channels) 크기의 입력 텐서를 가짐  
  MNIST에서는 이미지 포멧인 (28, 28, 1) 크기의 입력을 처리하도록 convnet을 설정해야 함  

#### 간단한 convnet 만들기
~~~
model = model.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28,  1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3 ,3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3 ,3), activation = 'relu'))

model.add(layers.flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))
~~~

- Conv2D, MaxPooling2D 층의 출력은 (height, width, channels) 크기의 3D 텐서
- 너비와 높이는 네트워크가 깊어질수록 작아지는 경향이 있음  
  --> 출력층에 가까워 질수록 너비와 높이가 작아진다는 말  
- model.add를 통해 층을 추가할 때, 순서가 꼬이지 않도록 조심 

#### 합성곱 연산
- dense layer와 합성곱 층 사이의 근본적인 차이  
  --> 완전 연결 층은 전역 패턴을 학습하는 반면에 합성곱 층은 지역 패턴을 학습  
- 이 핵심 특징은 convNet에 2가지 흥미로운 성질을 제공

1. 학습된 패턴은 평행 이동 불변성(translation invariant)을 가짐
- 예를 들어, 왼쪽 이미지 상단 모서리에서 어떤 패턴을 학습했으면, 오른쪽 상단 모서리에 동일한 패턴을 인식할 수 있음  
반면에 Dense Layer는 다르게 인식  
잘 생각해보면, 우리가 볼 때 평행 이동으로 인해 다르게 인식되지 않음  
--> 적은 샘플로 일반화 능력을 가질 수 있게 됨

2. 패턴의 공간적 계층 구조를 학습할 수 있음
- 첫 번째 합성곱 층이 edge같은 작은 지역 패턴을 학습하고,  
  두 번째 합성곱 층이 첫 번째 합성곱 층의 특성으로 구성된 더 큰 패턴을 학습  
  --> 이런 방식을 이용하여 convNet은 복잡하고 추상적인 시각적 개념을 효과적으로 학습  

- 합성곱 연산은 특성 맵(feature map) 이라고 불리우는 3D 텐서에 적용  
이 텐서는 2개의 공간 축(높이와 너비)과 깊이 축(채널 축)으로 구성  
-> RGB(컬러) 이미지는 channel이 3, 흑백은 1

- 합성곱 연산은 이러한 feature Map에서 작은 패치들을 추출하고  
  모든 패치에 같은 변환을 적용하여 출력 특성 맵(output feature map)을 만듬  

- output feature Map도 높이와 너비를 가지는 3D tensor  
  깊이(channel)은 층의 매개변수로 결정되기 때문에 그때 그때 다름  
  깊이 축은 더 이상 RGB를 나타내지 않음  
  일종의 filter 수를 나타냄 

- 합성곱 층에서 filter는 해당 층의 파라미터  
  Conv2D의 첫 번째 매개변수가 출력 특성 맵의 차원을 결정  
  필터는 입력 데이터의 어떤 특성을 인코딩한 결과  
  ex) 하나의 필터가 '입력에 얼굴이 있는지'를 인코딩 할 수 있음  

- model.add(layers.Conv2D(64 ( < -- ), (3, 3),  activation = 'relu'))  
  --> 64개의 filter가 존재  

- MNIST 예제에서 첫 번째 합성곱 층이 (28, 28, 1) 크기의 맵을 입력으로 받아 
  (26, 26, 32) 크기의 특성 맵을 출력함  
  --> 32개의 출력 채널은 각각 (26, 26) 크기의 배열을 가짐  
  이 값은 입력에 대한 필터의 응답 맵(response map)임  
  출력 맵 = 응답 맵

- 







### 용어 정리
- 패치(patch)  
  합성곱 연산에서 input/output에 해당하는 이미지 보드  
  
