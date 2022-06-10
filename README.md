# Trash Semantic Segmentation Competition

> [boostcamp AI Tech](https://boostcamp.connect.or.kr) - Level 2: CV_02 Bucket Interior

### Results

  * **Test dataset for public leaderboard**
    * mAP score: 0.7950
  * **Test dataset for private leaderboard**
    * mAP score: 0.7266

### Task

#### Semantic Segmentation Task Specifications

  * **주어진 쓰레기 사진에서 10종류의 쓰레기 객체 영역을 검출**
    * **Subtask 1: 픽셀 단위 개별 쓰레기 영역 검출**
      * Semantic mask suggestion
    * **Subtask 2: 검출된 영역의 쓰레기 분류**
      * Classes for each pixel

#### Image Dataset Specifications

  * **Dataset ratio**
    * Train & validation dataset: 79.98%
    * Test dataset for public leaderboard: 10.01%
    * Test dataset for private leaderboard: 20.02%
      * Public test dataset: 10.01%
      * Private test dataset: 10.01%

#### Main Difficulties

  * **Data imbalance**
    * 일부 유형의 쓰레기 사진의 비율이 유의미하게 낮았음
  * **Bounding box noise**
    * 많은 쓰레기 사진의 쓰레기 영역 annotation이 정확하지 않았음
  * **Tiny objects**
    * 대부분의 쓰레기 사진에 등장하는 쓰레기의 크기가 매우 작았음
    * 매우 많은 숫자의 매우 작은 쓰레기가 등장하는 사진이 유의미하게 많았음

### Approaches

  * **Selecting models**
    * 코드가 공개되어 있는 모델들 중 공식 performance가 높은 모델 위주로 사용
    * Ensemble 과정에서의 다양성을 위해 여러 모델을 실험에 사용
      * UperNet-BEiT-L
      * UperNet-Swin-L
      * HRNet
      * UNet
      * UNet++(decoder), EfficientNet_b4(encoder)
      * FPN(decoder), EfficientNet_b4(encoder)
      * DeepNet_resnet101
      * FCN_resnet50
  * **Increasing generalization performance**
    * 다양한 learning rate를 실험에 사용
      * 시작: 1e-4, 1e-6, 5e-5
      * 관리: fixed, cosine annealing, step decay
    * 다양한 optimizer를 실험에 사용: SGD, Adam, AdamW
    * 다양한 Loss를 실험에 사용: CrossEntropyLoss, FocalLoss
    * 다양한 Batch size를 실험에 사용: 2, 4, 8, 16
  * **Increasing the amount of training data**
    * Training data + Validation data 학습에 사용 (전체 데이터의 79.98%) 
    * Training data + Validation data + Public test data 학습에 사용 (Pseudo-labeling, 전체 데이터의 89.99%)

### Technical Specifications

> 가장 높은 mIoU score를 달성한 모델에 대해서만 기록

  * **Model: Pixel-wise hard voting ensemble**
    * UperNet-BEiT-L
    * UperNet-Swin-L

### Thoughts

> 이전 대회에서 한 차례 만났던 데이터셋이여서, EDA 및 데이터 분석에 대한 시간은 아끼고 modeling을 빠르게 시작하였다. 시간을 많이 확보한만큼 더 다양한 모델을 실험에 사용하였으며, 모든 모델에서 hyperparameter tuning, data augmentation 등 성능 개선을 위한 모든 시도를 적용해보았다. WandB와 Notion을 통해 서로의 진행상황과 결과를 원활하게 공유하였으며, 더 많은 ensemble의 시도와 새롭게 적용해본 pseudo labeling 등 지난 프로젝트들보다 훨씬 많은 시도들과 개선을 할 수 있었다. <br>
>
> <br>
>
> 이 대회에서는 모델링 과정에서 특히 더 많은 오류와 이슈들을 만났었지만, 팀원들끼리의 토론과 여러 시도들을 통해 모두 해결해나갔다. 결국 사용해보고자 했던 모델, 적용해보고자 했던 실험은 모두 마칠 수 있었던 성공적인 프로젝트였다. 우수 사례 발표를 들어보니 선정한 모델이 달랐을 뿐, 모든 솔루션이 그동안 토론했던 내용 중에 포함돼있었기 때문에, 그동안의 대회들을 거치면서 얻은 모든 팀원들의 성장을 체감할 수 있었다.
