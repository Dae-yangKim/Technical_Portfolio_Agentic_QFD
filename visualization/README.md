# visualization — QFD 산출물 시각화
서울과학기술대학교_데이터사이언스학과_25510099김대양_졸업심사_테크니컬포트폴리오

[`../pipeline`](../pipeline)이 만든 산출물 json 4개를 읽어 HoQ(House of Quality)와 각 행렬을 matplotlib으로 그리는 코드. `sample_input/`에 예시 입력이 들어 있어 clone 직후 바로 실행할 수 있다.

### 주요 특징

- **파이프라인 산출물을 그대로 입력**: `hoq.json`(Stage 1), `conflict_log.json`(Stage 2), `integrated_hoq.json`·`part_deployment.json`(Stage 3). 전처리나 별도 데이터 없음
- **그림 9종, 한 번 실행에 전부 생성**: CR 종류별 HoQ, Initial HoQ, Importance, Direction, EC Roof, N×N 충돌 행렬, Integrated HoQ, EC-PC 행렬, PC Roof. Separation으로 모델이 나뉜 경우 모델별로 그림
- **값에 따라 셀을 색으로 구분**: 관계 +9 진파랑 / +3 연파랑 / −3 연빨강 / −9 진빨강, Roof ±0.9 진한색 / ±0.5 연한색(초록 = 양, 주황 = 음), 충돌 셀 노랑, 판정 제외 셀 회색
- **Integrated HoQ에서 조정된 셀 강조**: Stage 2 Judge가 바꾼 Importance 셀을 굵은 테두리와 "전→후"(예: 3→0)로 표시해 무엇이 바뀌었는지 한 그림에서 보임
- **범례 없이 읽히는 라벨**: 행은 `CR-n · 고객 요구 문장`, 열은 `EC-n · 특성 이름(단위)`을 세로로 적음. 번호 순서는 파이프라인 목록 순서와 동일
- 순수 matplotlib. 셀 크기 고정, 행·열 수와 라벨 길이에 맞춰 그림 크기 자동 계산

### 설치

저장소 루트에서 환경을 만든다(루트 README 참고). 추가 설정이나 API 키는 필요 없다.

### 사용법

```bash
cd visualization
uv run python render.py                                      # sample_input → figures/ 에 PNG 14장
uv run python render.py --only initial_hoq,conflict          # 일부만
uv run python render.py --input ../pipeline/data/intermediate --out out   # 파이프라인을 직접 돌린 산출물로
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input` | `sample_input` | json 4개가 있는 폴더 |
| `--out` | `figures` | PNG 저장 폴더 |
| `--only` | 전부 | 그릴 그림 키 (쉼표 구분): `seg_hoq`, `initial_hoq`, `importance`, `direction`, `roof`, `conflict`, `integrated_hoq`, `ec_pc`, `pc_roof` |
| `--dpi` | `200` | PNG 해상도 |

파이프라인을 직접 돌린 경우 Stage 1·2 산출물은 `pipeline/data/intermediate/`, Stage 3 산출물은 `pipeline/data/output/`에 생기므로, json 4개를 한 폴더에 모아 `--input`으로 지정한다.

### 결과

`figures/`에 PNG가 생성된다. 예시 입력 기준 14장.

| 키 | 파일 | 내용 |
|---|---|---|
| `seg_hoq` | `seg_hoq_common.png`, `seg_hoq_professional.png`, `seg_hoq_casual.png` | CR 종류별 HoQ (관계 행렬 −9…+9, Priority) |
| `initial_hoq` | `initial_hoq.png` | 세 HoQ를 행 방향으로 쌓은 Initial HoQ |
| `importance` | `importance_matrix.png` | Importance Matrix (0/3/9) |
| `direction` | `direction_matrix.png` | Direction Matrix (↑/↓) |
| `roof` | `roof_matrix.png` | EC 간 상관 (±0.5 / ±0.9) |
| `conflict` | `conflict_matrix.png` | N×N 충돌 행렬 — 충돌 셀에 근거 EC 표기, 같은 종류 쌍은 × |
| `integrated_hoq` | `integrated_hoq_<model>.png` | 조정 반영 Integrated HoQ, 조정 셀 강조 (모델별) |
| `ec_pc` | `ec_pc_matrix_<model>.png` | EC-PC 관계 행렬 (0/3/9, 모델별) |
| `pc_roof` | `pc_roof_matrix_<model>.png` | PC 간 상관 (모델별) |

`<model>`은 `integrated_hoq.json`의 모델 이름(예: `professional_model`, `casual_model`)이며, 단일 모델이면 한 장씩 생성된다.
