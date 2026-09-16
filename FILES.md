# 파일별 역할

## 실행 스크립트 (`src/`)

| 파일 | 역할 |
|---|---|
| `run_dense.py` | ① Dense-only baseline 실행 (BiModernVBERT, 단일 subset) |
| `run_dense_all.py` | ①을 ViDoRe v1 10개 subset 전체에 대해 순회 실행, 결과를 `outputs/tables/main_results.csv`에 취합 |
| `run_vsplade_dense_sparse.py` | ①②④ 실행: Dense-only vs Global Sparse-only vs Dense+Global Sparse(RSF fusion) 비교, 단일 subset |
| `run_vsplade_dense_sparse_all.py` | 위를 10개 subset 전체에 대해 실행 (모델 1회 로딩 후 순회) |
| `run_vsplade_local_sparse.py` | ①③⑤ 실행: Dense-only vs Local Sparse-only vs Dense+Local Sparse 비교, 단일 subset |
| `run_vsplade_local_sparse_all.py` | 위를 10개 subset 전체에 대해 실행 |
| `run_vsplade_lexical_analysis_all.py` | RQ3: 전체 쿼리를 lexical-heavy/mixed/non-lexical로 자동 분류 후, 5가지 방식을 그룹별로 stratified 재평가 (10개 subset 전체) |
| `run_vsplade_bbox_grounding.py` | RQ4: ViDoRe v3의 bounding box 주석을 이용해, Local sparse가 고른 최고점 지역이 실제 정답 근거 위치와 얼마나 겹치는지 검증 (Idefics3 타일 분할 알고리즘을 직접 재구현해 픽셀 좌표 매핑) |
| `run_vsplade_local_finetune_experiment.py` | "Global로만 학습돼서 Local이 불리했다"는 가설을 검증하는 소규모 A/B fine-tuning 실험: 같은 조건에서 손실함수만 Global/Local로 다르게 한 LoRA 2개를 학습·비교 |

## 모델 래퍼 (`src/models/`)

| 파일 | 역할 |
|---|---|
| `bimodernvbert.py` | `ModernVBERT/bimodernvbert` dense 모델 래퍼. 이미지/쿼리를 단일 L2-정규화 벡터로 인코딩 |
| `vsplade.py` | `naver/v-splade-quality` sparse 모델 래퍼. 공식 API는 page-level(global) sparse 벡터만 제공하지만, 여기서는 pooling 이전 patch/타일 단위 표현을 직접 추출해 **global**(`encode_images`)과 **local**(`encode_images_local`, Idefics3 타일 단위) 두 가지 경로를 모두 제공 |

## 검색/점수 계산 (`src/retrieval/`)

| 파일 | 역할 |
|---|---|
| `dense_score.py` | Dense 점수 행렬 계산 (단일 벡터 코사인 유사도) |
| `sparse_score.py` | Local sparse 점수 계산 — 문서당 K개(가변) 지역 벡터 중 쿼리와 가장 잘 맞는 지역의 점수를 채택 (max, 절대 합산 금지) |
| `fusion.py` | Dense/Sparse 점수를 Relative Score Fusion(쿼리별 min-max 정규화 후 가중합)으로 결합 |

## 데이터 로딩 (`src/data/`)

| 파일 | 역할 |
|---|---|
| `load_vidore.py` | ViDoRe v1(BEIR 포맷)/v3(bounding box 포함) 데이터셋 로더 |
| `lexical_subset.py` | 쿼리를 lexical-heavy/mixed/non-lexical로 분류하는 정규식 기반 휴리스틱 (숫자, %/화폐, 대문자 약어, 코드형 토큰, entity 탐지) |

## 평가 (`src/evaluation/`)

| 파일 | 역할 |
|---|---|
| `metrics.py` | nDCG@k, Recall@k, MRR@k 등 검색 평가지표 (프레임워크 의존 없는 순수 함수) |
| `test_metrics.py` | `metrics.py`에 대한 단위 테스트 |

## 유틸리티 (`src/utils/`)

| 파일 | 역할 |
|---|---|
| `io.py` | 설정(config) 로딩, 실행 결과 디렉터리 생성/저장 헬퍼 |
| `logging.py` | 실행별 로그 파일 + 표준출력 동시 기록 로거 |

## 설정 (`configs/`)

| 파일 | 역할 |
|---|---|
| `dense.yaml` | `run_dense.py`/`run_dense_all.py`용 설정 (모델 이름, 배치 크기 등). `run_vsplade_*` 스크립트들은 이 설정 파일을 쓰지 않고 각 파일 상단에 상수로 값을 직접 명시함 |
