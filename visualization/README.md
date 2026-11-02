# visualization
- 서울과학기술대학교 데이터사이언스학과
- 25510099 김대양

**TL;DR: 파이프라인 산출물 json을 읽어 관계 행렬 · 상관 행렬 · 충돌 행렬 그림을 matplotlib으로 그리는 코드를 정리하고, 그림 구성 · 사용 라이브러리 · 사용법 · 결과를 이 문서에 구성했다.**

---

### 주요 특징

- 입력은 `pipeline/`이 만든 json 4개를 그대로 쓰도록 구성하였고, 별도 전처리나 데이터 변환 없이 산출물 폴더를 지정하면 그림이 생성되도록 코드를 구성하였다.
- 행렬은 matplotlib의 표 기능 대신 칸마다 사각형을 하나씩 그려서 만들었다. 칸 크기는 고정하고, 그림 전체 크기는 행 · 열 수와 글자 길이에 맞춰 코드가 계산한다. 그래서 행렬이 커지거나 라벨이 길어져도 코드를 수정할 필요 없이 수행 가능하다.
- 값에 따라 셀 색을 달리하였다. 관계 강도는 양 · 음을 파랑 · 빨강 계열의 두 단계 농도로, 상관은 초록 · 주황 두 단계로, 충돌 셀은 노랑, 판정 제외 셀은 회색으로 구분해 범례 없이 읽히도록 하였다.
- 조정 전후 비교가 필요한 행렬은 바뀐 셀에 굵은 테두리를 두르고 "전→후" 값을 함께 적어, 어느 셀이 어떻게 바뀌었는지 한 그림에서 보이도록 하였다.

### 사용 라이브러리

| 라이브러리 | 역할 |
|---|---|
| matplotlib | 셀 단위 사각형 그리기, 라벨 회전, PNG 저장 |
| json, argparse | 입력 로드, 실행 옵션 |

### 설치

저장소 루트의 README.md에서 안내된 내용대로 환경을 만들면 된다. 외부 API를 호출하지 않으므로 별도의 키 설정은 필요 없다.

### 사용법

```bash
python render.py                                      # sample_input → figures/ 에 PNG 14장
python render.py --only initial_hoq,conflict          # 일부만
```
옵션 정보는 `python render.py --help`로 확인할 수 있다.

### 결과

| 키 | 파일 | 내용 |
|---|---|---|
| `seg_hoq` | `seg_hoq_<종류>.png` | 요구사항 종류별 관계 행렬 |
| `initial_hoq` | `initial_hoq.png` | 종류별 행렬을 행 방향으로 쌓은 전체 관계 행렬 |
| `importance` | `importance_matrix.png` | 관계 강도 행렬 |
| `direction` | `direction_matrix.png` | 방향 행렬 |
| `roof` | `roof_matrix.png` | 기술 특성 간 상관 |
| `conflict` | `conflict_matrix.png` | 요구사항 간 충돌 행렬 |
| `integrated_hoq` | `integrated_hoq_<model>.png` | 조정 반영 통합 관계 행렬, 조정 셀 강조 |
| `ec_pc` | `ec_pc_matrix_<model>.png` | 기술 특성-부품 특성 관계 행렬 |
| `pc_roof` | `pc_roof_matrix_<model>.png` | 부품 특성 간 상관 |

> `<종류>`는 요구사항 종류(공통 · 사용 목적)이고, `<model>`은 integrated_hoq.json의 모델 이름이다. 모델이 나뉘지 않았으면 한 장씩 생성된다.
