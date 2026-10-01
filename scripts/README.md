# 실행·검증 도구

기본 대시보드, 공개 관측 3개, 예측·채밀 추천, 온도 실습 재학습은 저장소의 파일과 고정 TiRex 체크포인트만으로 실행한다. 원본 연구 CSV나 이전 개발 폴더는 필요하지 않다. 설치·실행은 루트 [README](../README.md)를 따른다.

| 도구 | 용도 |
|---|---|
| `fetch_model.py` | `models.lock.json`의 공식 체크포인트 다운로드와 크기·SHA-256 검사 |
| `check_bundle.py` | 초기 모델·관측·호환 레지스트리 검증. CI는 `--allow-missing-checkpoint` 사용 |
| `initialize_models.py` | 초기 모델을 실행 저장소에 복사하고 기존 활성 모델 보존 |
| `initialize_runtime.py` | 호환 레지스트리 초기화와 폴더 이동 시 로컬 MLflow 경로 갱신. 서버가 정지된 상태에서 실행 |
| `smoke_demo.py` | 실행 중인 서버의 읽기 전용 API 점검. `--train`을 명시하면 실습 데이터를 저장하고 실제 재학습 실행 |
| `generate_temperature_scenario.py` | 제공 합성 온도 실습 CSV 생성 기준 |
| `build_docs.py`, `build_docs_pdf.py` | 문서 생성. 별도 [문서 안내](../docs/README_docs.md) 참고 |

## 선택 사항: 과거 연구 벤치마크 재현

`benchmark_tirex2.py`, `prepare_additional_ufc_data.py`, `prepare_vecauce_data.py`는 과거 공개 데이터 비교 연구의 도구다. 기본 실행과 온도 실습에서는 호출하지 않는다. 보조 모듈을 가져오는 것만으로 데이터 가공·학습·다운로드가 실행되지 않는다.

이 저장소에는 가공한 시연 CSV와 출처·해시, 당시 [벤치마크 결과](../docs/evidence/tirex2_benchmark_20261001.json)를 포함한다. **전체 원본 연구 CSV와 이전 LSTM·Chronos 비교 입력 보고서는 포함하지 않는다.** 따라서 도구가 포함되어 있다는 것만으로 과거 벤치마크를 즉시 다시 실행할 수 있는 것은 아니다.

`benchmark_tirex2.py`에는 다음 인자가 필요하다.

| 인자 | 필요한 입력 |
|---|---|
| `--baseline-report` | 이전 비교 보고서 JSON. `origins`의 실제 LSTM·Chronos 예측과 정답, `settings.selected_context`, `provenance.snapshot_id`, `limitations` 포함 |
| `--data-root` | 아래 `raw/` 트리를 포함한 별도 데이터 디렉터리 |
| `--checkpoint` | `model.ckpt`와 `model-config.yaml`이 있는 로컬 TiRex-2 디렉터리 |
| `--worker-python` | `tirex-2`가 설치된 별도 Python 실행 파일 |
| `--revision` | 해당 체크포인트의 실제 공개 리비전 식별자 |
| `--output` | 결과 JSON 경로. 생략하면 저장소의 `evidence/tirex2_benchmark_20261001.json`; 같은 위치에 예측 `.npz`도 생성 |

필요한 원본 디렉터리 구성은 다음과 같다. 메타데이터에 기록된 원본 파일명과 공개 MD5·크기가 일치해야 하며, 이전 보고서의 데이터 스냅샷 해시와도 일치해야 한다.

```text
<data-root>/raw/
  zenodo_20399470_metadata.json
  dadosColmeia1Apis.csv
  dadosColmeia2Apis.csv
  DadosMeliponas.csv
  vecauce2021/
    metadata.json
    <메타데이터의 originalFileName에 해당하는 원본 CSV 5개>
    ReadMe.txt  # prepare_vecauce_data.py를 단독 실행할 때도 필요
```

UFC 원본은 [Zenodo 20399470](https://zenodo.org/records/20399470), Vecauce 원본은 [DataverseLV J1CQTJ](https://dv.dataverse.lv/dataset.xhtml?persistentId=doi:10.71782/DATA/J1CQTJ)에 공개되어 있다. 배포 CSV에 적용한 출처·시간 처리·가공 해시는 [관측 사례 manifest](../data/trend_cases/manifest.json)와 [공개 샘플 출처](../data/samples/사용법.md)에 보존했다. 재현 도구는 원본 다운로드를 대신하지 않으며, 두 독립 실행 보조 도구의 기본 원본 경로는 이 저장소의 `data/raw/`다. Vecauce 도구는 `--raw-directory`로 다른 원본 위치를 지정할 수 있다.
