# data_analysis_project

## 간단한 실행 사용 설명서
1. Anaconda를 설치하시오 (https://www.anaconda.com/download)
2. Conda Environment를 생성하세요 (`conda create --name "<YOUR ENV NAME>"`)
3. environment_bert_train.yml 파일 내의 dependency로 업데이트 하세요 (`conda env update --name bert_train --file environment_bert_train.yml`)
4. 최신 huggingface 라이브러리 따로 설치 (https://huggingface.co/docs/transformers/installation)
5. 학습 실행 (`python train_bert_mlm.py`)

## 학습 그래프
![image](https://github.com/agwaBom/data_analysis_project/assets/59073111/9dd8eea1-ab0e-491b-9c42-8a1058b50cbd)

## 세팅 변경 (Hyperparameter 등)
- `utils.py` 파일의 `def parse_args()` 함수를 참고하세요.
- bash 파일을 만들어서 여러 실험을 한꺼번에 돌릴 수 있게 구성이 가능합니다.

## 다른 모델로 바꾸고 싶다면...
- Huggingface에 배포되어있는 bert 모델들을 참고하세요 (https://huggingface.co/models)
- 현재는 bert-base-uncased를 기반으로 만들어져있습니다.

## 그 외의 정보...
- - 현재 epoch마다 valid set을 계산하여 최저 loss가 나온 것을 `./model_weight/bert_trained_{loss}`에 저장하고 있습니다. 이걸로 나중에 inference를 하면 됩니다.
- - KOLD 데이터 셋은 원 저자의 repo를 참고함 (https://github.com/boychaboy/KOLD/blob/main/data/kold_v1.json)
- 
