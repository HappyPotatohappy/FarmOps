# FarmOps

**FarmOps는 농업 현장의 데이터를 예측과 운영 의사결정으로 연결하는 프로젝트입니다. BeeOPS는 FarmOps의 양봉 운영 구현 사례입니다.**

이 저장소에는 BeeOPS의 실행 코드, 학습된 모델, 공개 관측·실습 CSV, 기획서와 시연 가이드가 함께 들어 있습니다. 현재 FarmOps의 양봉 분야 구현을 제공합니다.

BeeOPS는 벌통의 무게·외기온 관측을 바탕으로 **무게 예측, 채밀 추천, 온도 변화 감지, 앙상블 재학습**을 시연하는 프로젝트입니다. Windows와 Mac 팀원이 같은 소스와 같은 시작 모델로 실행하도록 구성했습니다.

이 폴더 자체가 GitHub 저장소의 루트입니다. `beeops_upgrade_lab`나 개인 컴퓨터의 다른 폴더를 가져올 필요가 없습니다. 초기 모델은 이 폴더를 만든 시점에 적용 중이던 **TiRex-2 + LSTM 공통 앙상블 v9**입니다. 모든 벌통이 같은 모델을 쓰며 벌통 선택에 따라 입력 관측이 바뀝니다. 상세 구성은 [bundle.json](bundle.json)에 기록했습니다.

## 팀원: 내려받아서 실행하기

**준비물:** Docker Desktop을 설치하고 실행합니다. Windows에서는 WSL 2 기반 Linux 컨테이너를 사용합니다. 아래 명령으로 받거나 GitHub의 **Code → Download ZIP**을 압축 해제해도 됩니다. 첫 빌드에는 인터넷과 충분한 디스크 공간이 필요합니다. Docker에 메모리 6GB 이상, 여유 디스크 15GB 이상을 권장합니다. Python, TensorFlow, CUDA를 호스트 컴퓨터에 따로 설치할 필요는 없습니다.

```sh
git clone https://github.com/HappyPotatohappy/FarmOps.git
cd FarmOps
```

ZIP으로 받았다면 압축을 푼 `FarmOps-main` 폴더를 사용합니다.

`compose.yaml`이 있는 이 폴더에서 터미널을 열고 실행합니다. Windows는 PowerShell, Mac은 터미널에서 같은 명령을 사용합니다.

```sh
docker compose up --build
```

서버 시작 로그가 나오면 **http://localhost:8012** 를 엽니다. 처음에는 라이브러리 설치와 모델 다운로드로 시간이 걸립니다. 다시 실행할 때는 빌드 캐시와 저장된 실행 데이터를 사용합니다.

```sh
# 백그라운드 실행
docker compose up -d --build

# 실행 상태와 로그
docker compose ps
docker compose logs -f beeops

# 종료: 관측·재학습 모델·작업 이력은 남습니다.
docker compose down
```

`down -v`는 저장 볼륨까지 지우는 명령이므로 평소 종료에 사용하지 않습니다. 기존 8010/8011 서비스와는 포트·저장소가 다릅니다. 다른 프로그램이 8012를 쓰면 루트에 `.env` 파일을 만들고 `BEEOPS_PORT=8014`를 적은 뒤 다시 실행하면 됩니다. 이 경우 브라우저 주소도 `http://localhost:8014`입니다.

## 시연 순서

1. 대시보드의 **현재 벌통**에서 브라질 Apis 2, Vecauce 3, 브라질 무침벌 1을 바꿉니다. 실제 관측과 과거 예측, 다음 1시간 예측, 앞으로 7일의 무게와 채밀 추천을 확인합니다.
2. **데이터 → 온도 변화 실습 → 02 · 고온 기간 → 파일 불러오기**를 누릅니다. 정상 기간 336시간과 고온 기간 168시간이 함께 준비됩니다.
3. **저장하고 재학습**을 누르면 `TEMP-DRIFT-PRACTICE`가 바로 선택되고 대시보드로 이동합니다. 관측 저장·준비 상태에서 실제 재학습 진행으로 이어집니다.
4. 대시보드에서 LSTM 학습 회차, 온도 변화 감지, 모델 등록·적용을 확인합니다. 재학습은 실제 LSTM 파라미터와 앙상블 결합 가중치를 갱신합니다. TiRex-2 본체는 유지합니다.
5. **앙상블 재학습 완료** 결과는 `확인`을 누르기 전까지 남습니다. 빠르게 끝나거나 새로고침해도 결과를 확인할 수 있습니다.
6. 다시 고온 파일을 불러와 저장하면 새로운 실제 재학습을 실행합니다. 관측은 중복 저장하지 않습니다. 같은 접수의 재전송이나 이미 실행 중인 같은 실습은 중복 학습하지 않습니다.
7. **03 · 후속 관측**을 저장해 추가 관측과 예측을 확인합니다. 운영 화면에서는 온도 기준과 재학습 이력을 볼 수 있습니다.

첫 실행은 공개 관측 사례 3개로 시작합니다. `TEMP-DRIFT-PRACTICE`는 실습을 저장할 때 추가됩니다. 고온 CSV는 합성 실습 데이터이며 공개 관측 사례와 구별합니다. 자세한 진행과 정상·실패 시 기대 결과는 [시연 가이드](docs/시연가이드.md), [온도 드리프트 실습](docs/온도_드리프트_실습.md)에 있습니다.

## 이 폴더에 들어 있는 것

| 경로 | 역할 |
|---|---|
| `app/` | FastAPI 서버, 예측·채밀 추천·데이터 검증·온도 감지·재학습 코드 |
| `app/static/` | 대시보드 HTML/CSS/JavaScript. 반복 재학습과 완료 알림 유지 기능 포함 |
| `data/trend_cases/` | 대시보드에 처음 표시하는 공개 관측 CSV 3개와 출처·구간 설명 |
| `data/simulations/temperature_practice/` | 정상·고온·후속·정상 대조군 CSV와 생성 기준 |
| `data/samples/` | 추가로 올릴 수 있는 공개 벌통 CSV와 출처 기록 |
| `data/horizon_models/9/` | 배포 시점의 LSTM, 정규화 설정, 결합 가중치, 평가, 학습 근거, TiRex 설정·라이선스 |
| `data/compatibility_seed/` | 기존 상태·버전 API가 필요로 하는 작은 공통 LSTM 레지스트리. 제품 예측 모델과는 별도 |
| `docs/` | 최신 기획서, 모델 상세 설명, 시연 가이드, 온도 실습 안내의 Markdown·HTML 원본 |
| `output/pdf/` | 기획서와 모델 상세 설명의 배포용 PDF |
| `scripts/` | 모델 다운로드·무결성 검사·독립 저장소 초기화·문서 생성·검증 도구. [도구별 설명](scripts/README.md) |
| `tests/` | 현재 기능과 패키지 초기화·경로 이동·다운로드의 회귀 테스트 |
| `Dockerfile`, `compose.yaml` | Windows·Mac 공통 CPU 실행 환경과 데이터 보존 볼륨 |
| `setup.sh`, `start.sh`, `verify.sh` | Docker 없이 Mac에서 설치·실행·검증할 때 사용하는 명령 |
| `requirements-*.txt` | 서비스와 TiRex 워커의 분리된 라이브러리 버전 |
| `models.lock.json` | 공식 모델의 고정 다운로드 주소·크기·SHA-256 |
| `bundle.json`, `FILE_MANIFEST.json` | 시작 모델과 폴더 구성·파일 검증 정보 |
| `.github/workflows/checks.yml` | GitHub에서 소스·패키지·화면 회귀 검사 자동 실행 |

실행하면 `runtime/` 또는 Docker의 `beeops-runtime` 볼륨에 관측, 업로드 작업, 온도 기준, MLflow 등록 이력, 새 모델이 생성됩니다. 최초 모델은 실행 저장소로 복사한 뒤 사용하므로 재학습해도 Git에 올린 시작 모델을 덮어쓰지 않습니다. 실행 데이터는 Git 업로드 대상이 아닙니다.

## 모델 파일을 GitHub로 공유하는 방법

약 381MB인 TiRex 체크포인트는 일반 GitHub 파일 제한보다 큽니다. **그 파일을 Git에 억지로 올리지 않아도 실행됩니다.** `.gitignore`에서 제외하며, Docker 빌드와 `setup.sh`가 [공식 고정 버전](https://huggingface.co/NX-AI/TiRex-2/tree/05e5b26db52bfb256f1ae1bdf785589850482de3)을 내려받습니다. 크기와 SHA-256이 현재 검증한 모델과 정확히 일치해야 설치가 완료됩니다. 토큰이나 API 키는 필요하지 않습니다.

직접 학습한 `lstm.keras`, 정규화 값, 결합 가중치, 평가 JSON은 작은 파일이므로 저장소에 들어갑니다. 다른 모델을 임의로 다시 학습해서 대체하지 않습니다. 로컬 최종 폴더에는 체크포인트도 이미 들어 있으며, GitHub에서 받은 소스에만 없을 수 있습니다. 처음 설치가 끝난 뒤 예측·재학습에는 외부 API를 호출하지 않습니다.

Mac 직접 실행 경로에서 공식 모델 주소에 접속할 수 없다면 기존 체크포인트를 `models.lock.json`의 `path` 위치에 복사한 뒤 `python scripts/check_bundle.py`로 검증할 수 있습니다. 실행 중인 컨테이너의 모델은 볼륨에 유지됩니다.

GitHub 파일 제한의 근거: [GitHub 대용량 파일 안내](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Docker 설치: [Docker Compose 설치 안내](https://docs.docker.com/compose/install/).

## 저장소 구조와 공유 범위

저장소 이름은 **FarmOps**, 현재 실행되는 서비스 이름은 **BeeOPS**입니다. 이 구현을 바로 실행할 수 있도록 `app/`, `data/`, `docs/`, `compose.yaml`을 저장소 루트에 배치했습니다. 앱의 BeeOPS 명칭과 API는 그대로 사용합니다.

코드, 학습된 LSTM, 제공 CSV, 최신 문서와 검사 도구를 함께 공유합니다. 개인 실행 환경인 `.venv`, `runtime`, `.env`와 큰 공식 `model.ckpt`는 `.gitignore`에서 제외합니다. 고정 모델 다운로드 정보는 `models.lock.json`에 들어 있어 Git LFS 설정 없이 실행할 수 있습니다.

## 팀 공동 작업과 데이터

코드와 문서는 GitHub로 공유하며 각자 내려받은 앱은 **각자의 로컬 데이터와 학습 이력**을 가집니다. GitHub에 코드를 올린다고 실행 중인 관측 DB가 실시간으로 공동화되지는 않습니다. 같은 시연을 재현하려면 같은 시작 모델과 제공 CSV를 사용합니다. 모두가 한 화면과 한 저장소를 동시에 사용하려면 별도 공용 서버에 이 앱을 띄우는 단계가 필요합니다.

팀의 코드 수정은 별도 브랜치에서 진행하고 PR로 합치는 방식을 권장합니다. 실행 중 생성된 DB·새 모델·캐시를 소스에 섞지 않습니다. 시작 모델을 교체하려면 해당 모델의 파일·인덱스·`bundle.json`·검증 근거를 함께 갱신해야 합니다.

## 검증하기

이미 실행한 컨테이너에서:

```sh
docker compose exec beeops /opt/venv-service/bin/python scripts/check_bundle.py
docker compose exec beeops /opt/venv-service/bin/python -m pytest tests -q -p no:cacheprovider
```

화면 회귀 검사는 호스트에 Node.js 20 이상이 있다면 실행할 수 있습니다. Python 패키지 설치 없이 Node 기본 기능만 사용합니다.

```sh
node --test tests/ui_automatic_recommendation.cjs tests/ui_dashboard_cases.cjs tests/ui_retraining_status.cjs tests/ui_practice_run.cjs
```

GitHub 자동 검사는 모델을 내려받지 않는 가벼운 소스·패키지·UI 검사입니다. 실제 ML 추론과 재학습 검증은 Docker 실행 후 시연 순서를 통해 확인합니다. 제공한 API 점검 도구도 사용할 수 있습니다.

```sh
# 관측 사례 3개의 예측과 문서 경로 확인
docker compose exec beeops /opt/venv-service/bin/python scripts/smoke_demo.py

# 현재 실행 저장소에 고온 관측을 저장하고 실제 재학습을 두 번 검증
docker compose exec beeops /opt/venv-service/bin/python scripts/smoke_demo.py --train --repeat 2
```

두 번째 명령은 로컬 실습 데이터와 모델 버전을 갱신합니다. 화면 알림은 데이터 탭의 버튼으로 실행하는 시연 순서에서 확인합니다. 실제 검증 결과는 [검증 기록](verification/RESULTS.md)과 [GitHub 다운로드 재현 검증](verification/GITHUB_DOWNLOAD.md)에 정리했습니다.

## Mac에서 직접 실행하기

직접 실행 경로는 macOS 14 이상인 Apple Silicon Mac, Python 3.11 기준입니다. Windows 팀원은 위 Docker 경로를 사용합니다.

```sh
sh setup.sh
sh start.sh
# http://127.0.0.1:8012
```

TensorFlow 서비스는 `.venv`, TiRex-2는 `.venv-tirex`를 사용합니다. NumPy 요구 버전이 달라 두 환경을 합치면 안 됩니다. 직접 설치에는 `uv` 또는 `python3.11`이 필요합니다. 가상환경은 다른 컴퓨터로 복사하지 말고 해당 컴퓨터에서 `setup.sh`로 만듭니다. `sh verify.sh`는 무결성·백엔드·UI 검사를 실행합니다.

## 문서 읽기

- [프로젝트 기획서](docs/BeeOPS_조별기획서.md) · [PDF](output/pdf/BeeOPS_조별기획서.pdf)
- [예측 모델 상세 설명](docs/BeeOPS_모델_상세설명.md) · [PDF](output/pdf/BeeOPS_모델_상세설명.pdf)
- [시연 가이드](docs/시연가이드.md)
- [온도 드리프트 실습](docs/온도_드리프트_실습.md)
- [문서 수정·HTML/PDF 재생성](docs/README_docs.md)

앱 메뉴의 **프로젝트 기획서·모델 설명서**에서도 같은 최신 문서를 읽을 수 있습니다. 코드의 입출력 명세는 실행 후 `/docs`에 있습니다.

모델 버전 번호의 증가는 학습·등록 이력이며 정확도 향상을 자동으로 보장하지 않습니다. 실습 반복 학습의 평가값은 같은 자료를 재사용한 설명용 수치입니다. 채밀 추천은 무게 예측을 이용한 운영 보조이며 꿀의 성숙도·수분 등 현장 확인을 대체하지 않습니다. 공개 데이터의 출처·라이선스와 TiRex LICENSE/NOTICE는 폴더에 함께 보존했습니다.
