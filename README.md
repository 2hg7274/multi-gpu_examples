# Multi-GPU

# 들어가며  

Large Language Model… 그야말로 Large Model의 시대라고 해도 과언이 아니겠다. ChatGPT, LLaMA 2, Mistral… 대용량 데이터와 거대한 모델들이 대거 나오고 있다. 

MNIST, CIFAR 같은 데이터를 처리하기에는 고사양의 GPU도 필요 없었을 것이다. 하지만 Large Model과 Large Data 학습이 등장하면서 고사양의 GPU의 니즈가 충분히 올라갔고, 여러 개의 GPU를 활용하려는 노력이 많아졌다. 

[Donut: Document Understanding Trasformer](https://arxiv.org/abs/2111.15664\) 을 학습시키기 위해 github에 올라온 [train.py](http://train.py) 코드를 분석 중 multi-gpu를 활용하여 학습하는 코드를 보게 되었고, 어떤 방식으로 multi-gpu 학습을 하는 것인지 궁금하게 되어 정리를 시작한다. (OOM 그만 보자!)

참고 사이트

- [https://pytorch.org/tutorials/beginner/ddp_series_theory.html](https://pytorch.org/tutorials/beginner/ddp_series_theory.html)
- [https://medium.com/daangn/pytorch-multi-gpu-학습-제대로-하기-27270617936b](https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b)
- [https://hongl.tistory.com/292](https://hongl.tistory.com/292)
- [https://medium.com/tesser-team/다중-gpu를-효율적으로-사용하는-방법-dp부터-fsdp까지-3057d31150b6](https://medium.com/tesser-team/%EB%8B%A4%EC%A4%91-gpu%EB%A5%BC-%ED%9A%A8%EC%9C%A8%EC%A0%81%EC%9C%BC%EB%A1%9C-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95-dp%EB%B6%80%ED%84%B0-fsdp%EA%B9%8C%EC%A7%80-3057d31150b6)

---

# 개념 & 용어 정리

- Single GPU vs Multi GPU : 1개의 GPU vs 2개 이상의 GPU
- GPU vs Node: Node는 1대의 컴퓨터를 말함.
- Single Node Single GPU: 1대의 컴퓨터에 1개의 GPU
- Single Node Multi GPU: 1대의 컴퓨터에 2개 이상의 GPU
- Multi Node Multi GPU: 여러 개의 컴퓨터에 여러 개의 GPU
- world size: 모든 노드에서 실행되는 총 프로세스 수 (예를 들어, 2대의 서버에 각각 4개의 GPU가 있다면 world size는 2*4=8)
- local world size: 각 노드에서 실행되는 총 프로세스 수 (위의 예를 이어서 하면 local world size는 4)
- rank: 프로세스 ID

![집합 통신](./doc_img/Untitled%2013.png)

집합 통신

---

# PyTorch Data Parallel

PyTorch에서 multi-gpu 학습을 위한 Data Parallel이라는 기능을 제공한다. 

[Optional: Data Parallelism — PyTorch Tutorials 2.2.0+cu121 documentation](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)

![Untitled](./doc_img/Untitled%2014.png)

딥러닝을 여러 개의 GPU에서 사용하려면 일단 모델을 각 GPU에 복사해서 할당해야 한다. 그리고 iteration을 할 때마다 batch를 GPU의 개수만큼 나눈다. 이렇게 나누는 과정을 **‘scatter’**한다고 하며 [실제로 Data Parallel에서 scatter 함수를 사용해서 이 작업을 수행한다.](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/data_parallel.py#L175) 이렇게 입력을 나누고 나면 각 GPU에서 forward 과정을 진행한다. 각 입력에 대해 모델이 출력을 내보내면 이제 이 출력들을 하나의 GPU로 모은다. 이렇게 tensor를 하나의 device로 모으는 것을 **‘gather’**이라고 한다. 

보통 딥러닝에서는 모델의 출력과 정답을 비교하는 Loss function이 있다. Loss function을 통해 loss를 계산하면 back-propagation을 할 수 있다. Back-propagation은 각 GPU에서 수행하며 그 결과로 각 GPU에 있던 모델의 gradient를 구할 수 있다. 만약 4개의 GPU를 사용한다면 4개의 GPU에 각각 모델이 있고 각 모델은 계산된 gradient를 가지고 있다. 이제 모델을 업데이트하기 위해 각 GPU에 있는 gradient를 또 하나의 GPU로 모아서 업데이트를 한다. 만약 Adam과 같은 optimizer를 사용하고 있다면 gradient로 바로 모델을 업데이트하지 않고 추가 연산을 한다. 

```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)
```

`nn.DataParallel`로 model을 감싸면 된다. replicate → scatter → parallel_apply → gather 순서대로 진행한다. Gather가 하나의 GPU로 각 모델의 출력을 모아주기 때문에 하나의 GPU 메모리 사용량이 많을 수 밖에 없다. 

DataParallel을 테스트하기 위한 코드는 다음과 같다.

```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
	model = nn.DataParallel(model)

model.to(device)

...

for i, (inputs, labels) in enumerate(trainloader):
	outputs = model(inputs)
	loss = criterion(outputs, labels)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
```

위 코드로 모델 학습 시, 0번 GPU가 다른 GPU에 비해 더 많은 메모리를 사용하는 것을 확인할 수 있다. 이렇게 하나의 GPU가 상대적으로 많은 메모리를 사용하면 batch_size를 많이 키울 수 없다. batch size는 학습 성능에 영향을 주는 경우가 많기 때문에 메모리 사용 불균형은 꼭 해결 해야하는 문제이다. 또한 학습을 더 빨리 하고 싶어 multi-GPU를 쓰는 경우도 많다. 학습이 오래 걸릴 경우 batch size 차이로 1주일을 더 학습 시켜야 하는 상황이 될 수도 있다.

# PyTorch Distributed Data Parallel

분산 학습 자체는 하나의 컴퓨터로 학습하는게 아니라 여러 컴퓨터를 사용해서 학습하는 경우를 위해 개발된 것이다. 하지만 multi-GPU 학습을 할 때도 분산 학습을 사용할 수 있다. 

DataPrarallel과 DistributedDataParallel의 가장 결정적인 차이는 **DataParallel은 single proces, multi-thread 방식으로 single node에서만 동작**하고 **DistributedDataParallel single/multi machine 여부 상관 없이 multi-process로 동작**한다는 점이다. 파이썬의 GIL로 인해 DataParallel이 DistributedDataParallel에 비해 더 느릴 수밖에 없다. 

> GIL (Global Interpreter Lock)이란?
파이썬의 표준 구현은 CPython이라고 한다. CPython은 소스 코드를 구문 분석해서 8비트 명령어로 이루어진 바이트코드로 변환하고, 스택 기반 인터프리터를 통해 바이트코드를 실행한다. 바이트코드 인터프리터에는 파이썬 프로그램이 실행되는 동안 일관성  있게 유지해야 하는 상태가 존재하는데 CPython은 Thread safe하지 않은 메모리 관리를 쉽게 하기 위해, GIL이라는 방법 으로 여러 개의 Thread 메모리 접근을 제한하는 형태로 일관성을 강제로 유지한다. 즉, 처음 여러 개의 Thread가 병렬로 존재한다고 하더라도 실제로는 특정 순간에 하나의 Thread만 동작한다는 것이다.
> 

DDP의 동작 과정에 대해서 알아보자.

![Untitled](./doc_img/Untitled%2015.png)

만약, Single node에 4개의 GPU가 있다고 해보자. 

각 GPU에 model과 optimizer를 복사해서 load한다. 초기 모델 매개변수가 동일할 뿐만 아니라 Optimizer도 동일한 무작위 seed를 사용한다. DDP는 교육 과정 전반에 걸쳐 내부적으로 이러한 동기화를 유지한다.

![Untitled](./doc_img/Untitled%2016.png)

그다음 `DataLoader`를 통해 InputBatch를 얻고, `DistributedSampler`를 이용한다. 

![Untitled](./doc_img/Untitled%2017.png)

`DistributedSampler`는 위 그림과 같이 각각의 GPU에 데이터를 분배하는 역할을 한다. 

![Untitled](./doc_img/Untitled%2018.png)

![Untitled](./doc_img/Untitled%2019.png)

![Untitled](./doc_img/Untitled%2020.png)

각 프로세스에서 모델은 서로 다른 입력을 받고 로컬에서 forward 및 backward 전달을 실행하며 입력이 다르기 때문에 지금 누적되는 gradient도 다르다. 이 시점에서 Optimizer 단계를 실행하면 장치 전체에 걸쳐 서로 다른 매개변수가 발생하고 결국 단일 분산 모델이 아닌 4개의 고유한 모델이 생성된다.

![Untitled](./doc_img/Untitled%2021.png)

DDP는 동기화 단계를 시작한다. 모든 복제본의 gradients는 버킷화된 **Ring-AllReduce Algorithm**을 사용하여 집계됩니다.

![Untitled](./doc_img/Untitled%2022.png)

그리고 이 알고리즘의 멋진 점은 기울기 계산과 통신이 겹친다는 것이다. 동기화 단계는 모든 기울기가 계산될 때까지 기다리지 않는다. 대신, backward pass가 계속 실행되는 동안 링을 따라 통신을 시작한다. 이를 통해 GPU가 항상 작동하도록 보장한다.

![Untitled](./doc_img/Untitled%2023.png)

이제 Optimizer 단계를 실행하면 모든 복제본의 매개변수가 동일한 값으로 업데이트된다. 시작했을 때 모든 프로세스의 복제본은 동일했고 이제는 끝까지 계속 동기화 상태를 유지한다.

### Ring All-Reduce Algorithm?

[What is Distributed Data Parallel(DDP)](https://velog.io/@kwjinwoo/What-is-Distributed-Data-ParallelDDP#why-you-should-prefer-ddp-over-dataparalleldp)

![Untitled](./doc_img/Untitled%2024.png)

![Untitled](./doc_img/Untitled%2025.png)

![Untitled](./doc_img/Untitled%2026.png)

![Untitled](./doc_img/Untitled%2027.png)

각각의 process 내 array를 subarray로 나눈다. 나누어진 subarray를 chunk라고 부르고 chunk[p]는 p번 째 chunk를 의미한다. 각 process의 chunk[p]를 다음 process로 보내고, 이전 process로 부터 chunk[p-1]를 받는다. 

이전 process로 부터 chunk[p-1]을 받은 각각의 process는 자신의 process에 있는 chunk[p-1]과 reduce을 진행하고 다음 process로 넘긴다.

recieve-reduce-send 과정을 p-1 번 반복하여 모든 process가 각 process의 다른 chunk를 얻도록 만들어 all-reduce algorithm을 마친다. 

# FSDP (Fully Sharded Data Parallel)

[Getting Started with Fully Sharded Data Parallel(FSDP)](https://tutorials.pytorch.kr/intermediate/FSDP_tutorial.html#how-to-use-fsdp)

PyTorch FSDP tutorials

DDP에서는 모델 파라미터, gradient, optimizer에서 사용하는 states 등을 모두 각 GPU에 보관하고 계산하는 데 사용한다. 만약 이들을 서로 다른 GPU에서 보관하고, 필요할 때만 다른 GPU에서 해당 states를 불러와서 사용한다면 어떨까? 토인에 대한 overhead가 늘어나는 대신, GPU가 다룰 수 있는 모델의 크기는 더 커질 수 있지 않을까? 

![Untitled](./doc_img/Untitled%2028.png)

DDP에서는 기본적으로 sampler를 통해 각 GPU에 서로 다른 데이터가 전송되며, 각 데이터를 이용해서 모델 파라미터의 gradients A, B, C, D를 계산한다. 이후, All Reduce 연산을 통해 gradients A, B, C, D에 대한 평균을 구한 뒤, 모든 GPU에 전달된다. 이후 Optimizer의 step을 통해 각 GPU에서 모델 파라미터가 업데이트 되고, 똑같은 gradients 값을 사용했기 때문에, 똑같은 모델 정보가 보장된다.

반면, FSDP에서는 모델의 모든 정보가 하나의 GPU에 있는 것이 아니라, 여러 GPU에 분산되어(sharded) 있다. 따라서 forward 과정에서 모델의 각 layer를 통과할 때마다 다른 GPU에 저장되어 있는 파라미터를 가져와 사용하고 제거한다 (All Gather 연산). 이후 backward 과정에서 다시 gradients를 계산하기 위해 다른 GPU에 저장되어 있는 파라미터를 가져와서 사용하고 (All Gather 연산), 각 GPU에서 계산된 gradients를 다시 원래 속해 있던 GPU에 전달하기 위해서 Reduce Scatter 연산을 사용한다. 

최종적으로 각 GPU에는 각 GPU가 갖고 있던 모델에 대한 gradients만 남기 때문에, 이후 Optimizer의 step 연산을 통해 모델의 파라미터를 업데이트할 수 있다. 

FSDP는 모델만 분산하여 각 GPU에 저장할 뿐, 기본적인 구조는 DDP와 매우 유사하다. 

그렇다면, FSDP가 DDP보다 epoch 당 학습 속도가 더 빠를까? 

실제로는 그렇지 않을 수도 있다. DDP와 달리, FSDP에서는 GPU 사이의 데이터 전송이 더 빈번하게 이루어지기 때문에, 이로 인해 기존의 DDP보다 더 느릴 수도 있다. multi-nodes라면 서버 간의 통신 속도까지 고려하여 FSDP가 DDP보다 더욱 느려질 수도 있다.