# 외부 모델·데이터 출처

## TiRex-2

- 제공: NXAI / NX-AI
- 모델: https://huggingface.co/NX-AI/TiRex-2
- 고정 리비전: `05e5b26db52bfb256f1ae1bdf785589850482de3`
- 라이선스: Apache License 2.0
- 원문: [LICENSE](data/horizon_models/9/tirex2/LICENSE), [NOTICE](data/horizon_models/9/tirex2/NOTICE)
- 체크포인트는 변경하지 않았습니다. BeeOPS에서 LSTM과 결합하고 입력·출력 변환 및 결합 가중치 학습을 수행합니다. 파일 크기·SHA-256은 `models.lock.json`에 있습니다.

## 공개 벌통 관측

대시보드의 세 사례는 아래 공개 자료를 시간별로 집계하고 특정 연속 구간을 추출한 파생 자료입니다. 무게·외기온은 해당 시간 구간의 중앙값을 사용하고 결측을 임의로 보간하지 않았습니다. 원본 시계의 시간대가 명확하지 않아 CSV의 `+00:00`은 기록상 가정입니다. 원본 시간대가 확인되었다는 의미가 아닙니다. 무게 급변의 사건 표시는 원본 현장 작업의 확정 정답이 아닙니다.

- **UFC Apiary, Brazil**: JUCÁ DOS SANTOS, FELIPE; abrahao bomfim, isac gabriel; G. Gomes, Danielo; Coelho, Alexandre; Rafael Braga, Antonio; Cavalcante, Marcelo Casimiro; Freitas, Breno M. *Hive Monitoring (Apis mellifera and Stingless Bees) and Rainfall Data – Federal University of Ceará (UFC) Apiary – Brazil*. DOI: [10.5281/zenodo.20399470](https://doi.org/10.5281/zenodo.20399470). CC BY 4.0.
- **Vecauce, Latvia**: Zacepins, Aleksejs; Komasilova, Olvija; Komasilovs, Vitalijs (2025). *Bee colony monitoring data in Vecauce, Latvia, summer 2021*. DataverseLV, V1. DOI: [10.71782/DATA/J1CQTJ](https://doi.org/10.71782/DATA/J1CQTJ). CC BY 4.0.

원본 URL, 파일 해시, 필터링·집계 내용과 사례 범위는 `data/trend_cases/manifest.json`, `data/real_hive.provenance.json`, `data/samples/*.provenance.json`에 보존했습니다. 그 안의 `data/raw/...` 경로는 전처리 당시 원본 파일을 가리키는 기록이며, 실행에는 이미 처리된 CSV를 사용합니다. 전체 원자료는 위 공개 저장소에서 확인할 수 있습니다.

CC BY 4.0 원문: https://creativecommons.org/licenses/by/4.0/

## 합성 실습 자료와 Python 라이브러리

`data/simulations/temperature_practice/`는 BeeOPS 온도 변화·재학습 기능을 시연하기 위해 생성한 합성 자료입니다. 실제 양봉장 관측으로 표시하지 않습니다.

Python 라이브러리는 각 배포 패키지의 라이선스를 따릅니다. 서비스와 TiRex 워커는 별도 환경에 설치하며 사용 버전은 `requirements-*.txt`에 기록했습니다. 이 안내는 팀 자체 코드에 새로운 오픈소스 라이선스를 부여하지 않습니다.
