from dotenv import load_dotenv
load_dotenv()

MODEL = "gpt-5.1"
SERVICE_TIER = "auto"

SEGMENT_A = "Professional"
SEGMENT_B = "Casual"

MAX_DEBATE_ROUNDS = 3  # 토론 최대 라운드 (EC·Roof·Matrix Fill·Advocate·PC·Target Value 토론 공통)

RAW_DATA_PATH     = "data/raw/reddit_reviews.xlsx"
INTERMEDIATE_PATH = "data/intermediate"
OUTPUT_PATH       = "data/output"
CHROMA_BASE_PATH  = "data/chroma"

CHROMA_COLLECTION = "laptop_cr"

# ── Stage 0 ──────────────────────────────────────────────────────────────────
REUSE_TAGGED_REVIEWS           = True  # tagged_reviews.xlsx가 있으면 Phase 1(태깅)을 건너뛰고 재사용
MAX_CRS_PER_SEGMENT            = 13    # 세그먼트당 CR 추출 상한 (추출 프롬프트에 전달; HoQ 행 수를 20개 안팎으로 유지)
MIN_REVIEW_SUPPORT             = 2     # CR당 최소 evidence 리뷰 수
MAX_REVIEWS_PER_CHUNK          = 200   # 컨텍스트 초과 시 청크 크기
CANDIDATE_SIM_THRESHOLD        = 0.70  # Common CR 후보 탐색용
INTRA_CR_DEDUP_THRESHOLD       = 0.90  # Common CR 내부 중복 "후보" 탐색용 (판정은 LLM)
MAX_WORKERS                    = 10    # ThreadPoolExecutor

# ── Stage 1 ──────────────────────────────────────────────────────────────────
MAX_ECS                        = 18    # EC 개수 상한

# ── Stage 2 ──────────────────────────────────────────────────────────────────
ROOF_THRESHOLD                 = -0.5  # τ_R: Roof 충돌 강도 임계값 — |cor| > |τ_R| 인 강상관(±0.9)만 간접 충돌 후보
MAX_JUDGE_ROUNDS               = 3     # Judge Phase 최대 토론 라운드

# ── Stage 3 ──────────────────────────────────────────────────────────────────
MAX_PCS                        = 25    # PC 개수 상한
DROP_ZERO_IMPORTANCE_PC        = True  # Part Importance = 0 인 PC를 최종 산출에서 제외