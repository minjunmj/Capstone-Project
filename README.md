# Local vs Global Sparse Retrieval for Visual Document Retrieval

Visual Document Retrieval(문서 이미지 검색)에서 **"문서를 여러 지역(local
region)으로 나눠 매칭하는 방식(Local Sparse)이, 문서 전체를 하나의 벡터로
매칭하는 방식(Global Sparse)보다 검색 성능이 더 좋을 것"**이라는 가설을
검증한 연구입니다.

> 이 저장소는 두 번째(모델을 교체한) 실험 단계부터의 코드만 담고 있습니다.
> 첫 번째 실험(ColQwen2.5 기반)은 별도로 진행됐으며 여기 포함되지 않았습니다.

## 한 줄 결론

**가설은 기각되었습니다.** ViDoRe v1 벤치마크 10개 subset 전부에서 Global
Sparse가 Local Sparse를 이겼습니다(검색 순위 기준). 다만 Local Sparse는
**"실제 정답 근거가 문서의 어느 위치에 있는지 찾아내는 능력"**에서는 랜덤보다
뚜렷이 우수했습니다 — 검색 순위와 위치 파악력은 서로 다른 능력이라는 것이
이 연구의 핵심 발견입니다. 자세한 수치와 해석은 [`RESULTS.md`](./RESULTS.md)를
참고하세요.

## 사용 모델

| 역할 | 모델 | 비고 |
|---|---|---|
| Dense (검색) | [`ModernVBERT/bimodernvbert`](https://huggingface.co/ModernVBERT/bimodernvbert) | 250M, 단일벡터 bi-encoder |
| Sparse (검색) | [`naver/v-splade-quality`](https://huggingface.co/naver/v-splade-quality) | 같은 백본 계열 위에 SPLADE 방식으로 실제 학습된 모델 |

두 모델 모두 PyPI에 배포되지 않은 연구용 코드가 필요합니다 (`colpali_engine`의
`vbert` 브랜치, `naver/v-splade` 학습 저장소). 아래 "환경 설정"을 꼭 먼저
읽어주세요.

## 실험 구성 (5가지 방식)

| # | 방식 | 설명 |
|---|---|---|
| ① | Dense-only | BiModernVBERT 단독 |
| ② | Global Sparse-only | V-SPLADE, 문서 전체를 하나의 sparse 벡터로 pooling |
| ③ | Local Sparse-only | V-SPLADE, 문서를 타일(지역) 단위로 나눠 각각 sparse 벡터 생성 |
| ④ | Dense + Global Sparse | ①②를 Relative Score Fusion으로 결합 |
| ⑤ | Dense + Local Sparse | ①③을 Relative Score Fusion으로 결합 |

추가로 두 가지 후속 실험을 진행했습니다:
- **RQ4 (bbox grounding)**: Local sparse가 고른 최고점 지역이 실제 정답 근거
  위치(ViDoRe v3의 bounding box 주석)와 얼마나 겹치는지 검증
- **Local-objective fine-tuning**: "Global로만 학습돼서 Local이 불리했다"는
  가설을 직접 검증하기 위해, 같은 조건에서 손실함수만 Local로 바꿔 소규모
  LoRA fine-tuning 실험 진행

## 환경 설정

```bash
# 1. 기본 의존성
pip install -r requirements.txt

# 2. BiModernVBert 클래스가 담긴 colpali_engine의 vbert 브랜치 (PyPI 미배포)
git clone --branch vbert --single-branch https://github.com/illuin-tech/colpali.git third_party/colpali
cd third_party/colpali && pip install -e . --no-deps && cd -

# 3. V-SPLADE 학습/추론 코드 (PyPI 미배포)
git clone https://github.com/naver/v-splade.git third_party/v-splade
```

> **주의**: 위 두 저장소의 코드에는 실제 체크포인트 로딩 버그가 여러 건
> 있었고, 이 연구 과정에서 로컬로 패치해서 사용했습니다 (체크포인트마다
> 내부 레이어 구조 규약이 다른 문제, vocab 크기 불일치 등). 패치 내용과
> 이유는 원본 연구 로그에 상세히 기록돼 있습니다. 이 저장소를 그대로
> clone한 `third_party/`에는 이 패치가 적용되어 있지 않으므로, 스크립트를
> 그대로 실행하면 같은 버그를 다시 만날 수 있습니다.

`src/models/vsplade.py`의 `V_SPLADE_REPO` 상수가 `/workspace/third_party/v-splade`로
하드코딩되어 있으니, 다른 경로에 clone했다면 이 값을 맞춰줘야 합니다.

## 실행 방법

```bash
# ① Dense-only baseline (BiModernVBERT)
python src/run_dense.py --config configs/dense.yaml --subset tatdqa
python src/run_dense_all.py   # ViDoRe v1 10개 subset 전체

# ②④ Dense + Global Sparse
python src/run_vsplade_dense_sparse.py --subset tatdqa
python src/run_vsplade_dense_sparse_all.py   # 전체 subset

# ③⑤ Dense + Local Sparse
python src/run_vsplade_local_sparse.py --subset tatdqa
python src/run_vsplade_local_sparse_all.py   # 전체 subset

# Lexical-heavy 쿼리만 따로 재평가 (RQ3)
python src/run_vsplade_lexical_analysis_all.py

# RQ4: bbox grounding (ViDoRe v3)
python src/run_vsplade_bbox_grounding.py --subset finance_en --n-samples 999999

# Local-objective fine-tuning 실험
python src/run_vsplade_local_finetune_experiment.py
```

## 문서 안내

- [`RESULTS.md`](./RESULTS.md) — 모든 실험의 실측 결과와 해석
- [`FILES.md`](./FILES.md) — 각 파일의 역할 정리
