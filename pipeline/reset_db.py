"""ChromaDB 초기화 — Stage 0을 다시 돌리기 전에 실행 (리뷰·CR 벡터 DB를 전부 비운다)"""
import os
import shutil

import settings

base = settings.CHROMA_BASE_PATH
if not os.path.isdir(base):
    print(f"없음 (스킵): {base}")
else:
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  삭제 완료: {path}")

print("\nChromaDB 초기화 완료")
