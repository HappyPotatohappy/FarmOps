# 문서 수정과 다시 만들기

문서 내용은 이 폴더의 Markdown 4개가 원본입니다. HTML과 PDF를 직접 고치지 않고 원본을 수정한 뒤 생성합니다. 앱 실행에는 문서 생성용 패키지가 필요하지 않습니다.

## HTML 4종 생성

저장소 루트에서 Python 3.10 이상으로 실행합니다. 별도 패키지 설치나 원래 개발자의 폴더는 필요하지 않습니다.

```sh
python scripts/build_docs.py
```

Mac에서 `python` 명령이 없으면 `python3`를 사용합니다. 결과는 `docs/`의 기획서·모델 상세 설명·온도 실습·시연 가이드 HTML입니다. 그림과 CSS는 HTML 안에 포함됩니다. 지원하는 Markdown은 제목, 문단, 목록, 표, 코드, 인용문, 이미지입니다. 원문 SHA-256과 생성 시각도 기록합니다. 동일 생성 시각까지 재현하려면 `SOURCE_DATE_EPOCH` 환경변수에 고정 Unix 시간을 지정합니다.

기획서와 모델 안내는 서버의 `/proposal`, `/model-guide`에서도 열립니다. 보조 가이드 2개는 로컬 HTML로 엽니다. 로컬 파일에서는 문서 간 링크가 HTML 파일로 연결되고, 서버에서는 등록되지 않은 보조 문서 주소를 링크로 만들지 않습니다. 구현 근거 파일은 Markdown의 상대 링크에서 확인합니다.

## PDF 저장: Windows·Mac 공통

생성된 HTML을 브라우저에서 열고 **인쇄 또는 PDF 저장** 버튼을 누릅니다. 인쇄 대상은 PDF, 용지는 A4, 배율은 100%로 설정하고 브라우저 머리글·바닥글은 끕니다. 표 배경을 보존하려면 배경 그래픽을 켭니다. 기획서와 모델 안내를 아래 이름으로 저장합니다.

- `output/pdf/BeeOPS_조별기획서.pdf`
- `output/pdf/BeeOPS_모델_상세설명.pdf`

한글 글꼴은 Windows의 맑은 고딕, macOS의 Apple SD Gothic Neo, Linux의 Noto Sans CJK KR 또는 Noto Sans KR을 사용합니다. 이 저장소는 운영체제의 상용 글꼴 파일을 복사·배포하지 않습니다. 글꼴과 인쇄 엔진에 따라 페이지 나눔이 달라질 수 있으므로 저장 후 한글, 표, 코드, 그림이 잘리지 않았는지 확인합니다.

## 선택: 명령으로 PDF 2종 생성

별도의 문서용 Python 환경에서 다음 명령을 사용합니다. 이 경로는 WeasyPrint와 시스템 글꼴·Pango 라이브러리가 필요하며 서비스용 Docker 이미지에 포함되지 않습니다. 기본 팀 실행과 시연은 이 도구를 설치하지 않아도 됩니다.

```sh
python -m pip install -r scripts/requirements-docs.txt
python scripts/build_docs_pdf.py
```

시스템 라이브러리가 없다는 오류가 나면 브라우저 PDF 저장을 사용하거나 [WeasyPrint 공식 설치 안내](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation)에 맞춰 환경을 준비합니다. macOS Homebrew에서 라이브러리를 찾지 못하는 경우 `DYLD_FALLBACK_LIBRARY_PATH`에 Homebrew의 `lib` 경로를 지정할 수 있습니다. Linux에서는 한국어가 포함된 Noto 글꼴을 설치해야 합니다.

PDF 스크립트는 먼저 최신 HTML을 만들고 두 본문 PDF만 저장합니다. `--output-dir`로 다른 출력 폴더를 지정할 수 있습니다. 브라우저 자동화, CDP, 제품 서버, 모델·관측 저장소에 접근하지 않습니다. 이번 배포 PDF는 WeasyPrint 70.0과 Poppler로 생성·검수했으며 상세 기록은 `evidence/document_audit.json`에 있습니다.
