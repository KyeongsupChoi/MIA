    ├── data
    │   ├── external                <- 타사 소스의 데이터.
    │   ├── interim                 <- 변환된 중간 데이터.
    │   ├── processed               <- 모델링을 위한 최종 데이터 세트.
    │   └── raw                     <- 원본 데이터.
    |
    ├── docs                        <- 기본 Sphinx 프로젝트; 자세한 내용은 sphinx-doc.org 참조
    │
    ├── models                      <- 훈련된 모델, 모델 예측 또는 모델 요약
    │   └── Carmine400i70a.h5       <- 후방-전방 흉부 엑스레이에서 폐렴 감지 70% 정확도
    │
    ├── notebooks                   <- Jupyter 노트북. 이름 규칙은 번호(정렬을 위해),
    │                                   작성자의 이니셜, 그리고 짧은 설명, 예:
    │                                   `1.0-jqp-initial-data-exploration`.
    │
    ├── references                  <- 데이터 사전, 설명서 및 기타 모든 설명 자료.
    │   └── research_papers         <- PDF 형식의 학술 연구 논문
    │
    ├── reports                     <- HTML, PDF, LaTeX 등으로 생성된 분석
    │   └── figures                 <- 보고서에 사용될 그래픽 및 그림
    |
    ├── src                         <- 이 프로젝트에서 사용되는 소스 코드.
    │   ├── __init__.py             <- src를 Python 모듈로 만듦
    │   │
    │   ├── data                    <- 데이터 다운로드 또는 생성 스크립트
    │   │   └── make_dataset.py     <- 원본 데이터를 모델링을 위한 특징으로 변환하는 스크립트
    │   │
    │   ├── features                <- 원본 데이터를 모델링을 위한 특징으로 변환하는 스크립트
    │   │   ├── build_features.py   <- 폐렴 진단을 위한 800개의 이미지 데이터 세트를 생성하는 스크립트
    │   │   └── create_sliced.py    <- 슬라이스된 CSV 데이터 세트를 생성하는 스크립트
    │   │
    │   ├── models                  <- 모델을 훈련하고 훈련된 모델을 사용하여 예측하는 스크립트
    │   │   │
    │   │   ├── predict_model.py    <- 이미지를 통해 폐렴을 예측하기 위해 Carmine400 모델을 사용하는 스크립트
    │   │   └── train_model.py      <- 흉부 엑스레이에서 폐렴을 감지하기 위해 400개의 이미지를 사용하여 Carmine400 모델을 훈련하는 스크립트
    │   │
    │   ├── tests                   <- 자동화된 테스트를 실행하는 스크립트
    │   │   ├── carmine_test.py     <- Carmine 모델을 테스트하는 스크립트
    │   │   └── tangerine.py        <- Tangerine 모델을 테스트하는 스크립트
    │   │
    │   └── visualization           <- 탐색적 및 결과 지향 시각화를 생성하는 스크립트
    │       ├── exploratory.py      <- 탐색적 데이터 분석을 수행하는 스크립트
    │       ├── image_count.py      <- "raw/img" 폴더 내 이미지 수를 세는 스크립트
    │       └── visualize.py        <- OpenCV를 사용하여 흉부 엑스레이를 시각화하는 스크립트
    │
    ├── LICENSE
    ├── Makefile                    <- `make data` 또는 `make train`과 같은 명령을 포함한 Makefile
    ├── README.md                   <- 이 프로젝트를 사용하는 개발자를 위한 최상위 README.
    │
    │
    ├── requirements.txt            <- 분석 환경을 재현하기 위한 요구 사항 파일, 예:
    │                                   `pip freeze > requirements.txt`로 생성
    │
    ├── setup.py                    <- 프로젝트를 pip 설치 가능하게 함 (pip install -e .)으로 src를 임포트 가능하게 만듦
    |
    └── tox.ini                     <- tox 실행 설정을 포함한 tox 파일; 자세한 내용은 tox.readthedocs.io 참조
