    ├── data
    │   ├── external                <- 타사 소스의 데이터.
    │   ├── interim                 <- 변환된 중간 데이터.
    │   ├── processed               <- 모델링을 위한 최종 데이터 세트.
    │   └── raw                     <- 원본 데이터.
    |
    ├── docs                        <- 기본 Sphinx 프로젝트; 자세한 내용은 sphinx-doc.org 참조
    │
    ├── models                      <- 훈련된 모델, 모델 예측 또는 모델 요약
    │   └── final_model.keras       <- 2단계 ResNet50 전이 학습 모델
    │
    ├── notebooks                   <- Jupyter 노트북. 프로토타이핑 및 EDA용
    │   └── Carmine400train.ipynb   <- ResNet50 훈련 데모 노트북
    │
    ├── references                  <- 데이터 사전, 설명서 및 기타 모든 설명 자료.
    │   └── research_papers         <- PDF 형식의 학술 연구 논문
    │
    ├── reports                     <- HTML, PDF, LaTeX 등으로 생성된 분석
    │   └── figures                 <- 보고서에 사용될 그래픽 및 그림
    |
    ├── src                         <- 이 프로젝트에서 사용되는 소스 코드.
    │   ├── __init__.py             <- src를 Python 모듈로 만듦
    │   ├── config.py               <- 공유 상수 및 설정
    │   │
    │   ├── data                    <- 데이터 다운로드 또는 생성 스크립트
    │   │   └── make_dataset.py     <- TSV→CSV 필터링, 균형화, train/val/test 분할
    │   │
    │   ├── features                <- 원본 데이터를 모델링을 위한 특징으로 변환하는 스크립트
    │   │   ├── build_features.py   <- 이미지 로딩, 정규화, 클래스 가중치 계산
    │   │   └── create_sliced.py    <- 대체 CSV 분할 유틸리티
    │   │
    │   ├── models                  <- 모델을 훈련하고 예측하는 스크립트
    │   │   ├── predict_model.py    <- 흉부 엑스레이 단일 및 배치 추론
    │   │   └── train_Carmine400.py <- 교차 검증을 포함한 2단계 전이 학습 트레이너
    │   │
    │   └── visualization           <- 탐색적 및 결과 지향 시각화를 생성하는 스크립트
    │       ├── exploratory.py      <- 원시 TSV 데이터셋에 대한 EDA
    │       ├── gradcam.py          <- Grad-CAM 설명 가능성 오버레이
    │       ├── image_count.py      <- 이미지 파일 카운터 유틸리티
    │       └── visualize.py        <- OpenCV를 사용하여 흉부 엑스레이를 시각화하는 스크립트
    │
    ├── tests                       <- 단위 테스트 (pytest)
    │   ├── test_build_features.py
    │   ├── test_config.py
    │   ├── test_make_dataset.py
    │   └── test_predict.py
    │
    ├── Dockerfile                  <- 재현 가능한 컨테이너화 환경
    ├── LICENSE
    ├── Makefile                    <- `make data` 또는 `make train`과 같은 명령을 포함한 Makefile
    ├── README.md                   <- 이 프로젝트를 사용하는 개발자를 위한 최상위 README.
    ├── requirements.txt            <- 분석 환경을 재현하기 위한 고정된 종속성
    ├── setup.py                    <- 프로젝트를 pip 설치 가능하게 함 (pip install -e .)
    └── tox.ini                     <- tox 실행 설정을 포함한 tox 파일; 자세한 내용은 tox.readthedocs.io 참조
