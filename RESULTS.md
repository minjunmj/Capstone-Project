# 연구 결과

## 1. 5가지 방식 비교 (ViDoRe v1, 10개 subset 평균 nDCG@5)

| # | 방식 | nDCG@5 |
|---|---|---|
| ① | Dense-only (BiModernVBERT) | 0.377 |
| ② | Global Sparse-only (V-SPLADE) | 0.694 |
| ③ | Local Sparse-only (V-SPLADE, 타일 단위) | 0.643 |
| ④ | **Dense + Global Sparse (RSF fusion)** | **0.714 (1위)** |
| ⑤ | Dense + Local Sparse (RSF fusion) | 0.669 |

subset별 상세:

| subset | ① Dense | ② Global | ③ Local | ④ Dense+Global | ⑤ Dense+Local |
|---|---|---|---|---|---|
| syntheticDocQA_healthcare | 0.4937 | 0.9426 | 0.8784 | 0.9389 | 0.9050 |
| tabfquad | 0.4775 | 0.5421 | 0.4215 | 0.6200 | 0.5472 |
| syntheticDocQA_ai | 0.4727 | 0.9285 | 0.8804 | 0.9435 | 0.8659 |
| syntheticDocQA_energy | 0.4494 | 0.8710 | 0.7999 | 0.8783 | 0.8144 |
| infovqa | 0.4352 | 0.8007 | 0.7796 | 0.8156 | 0.7886 |
| syntheticDocQA_gov_reports | 0.4129 | 0.8867 | 0.8386 | 0.8982 | 0.8564 |
| arxivqa | 0.4081 | 0.6691 | 0.6574 | 0.6892 | 0.6716 |
| tatdqa | 0.2756 | 0.6148 | 0.5366 | 0.6033 | 0.5491 |
| shiftproject | 0.1930 | 0.2096 | 0.1752 | 0.2863 | 0.2368 |
| docvqa | 0.1516 | 0.4758 | 0.4639 | 0.4620 | 0.4497 |

**핵심 결과: ②vs③(순수 sparse), ④vs⑤(fusion 이후) 둘 다 10개 subset 전부
Global이 Local을 이김.** 예외 없음.

## 2. Lexical-heavy 쿼리만 따로 봤을 때 (RQ3)

가설의 핵심은 "숫자/약어/ID 같은 lexical 단서가 여러 개 몰려있는 질의
(lexical-heavy)에서는 Local이 유리할 것"이었다. 자동 라벨러로 전체 쿼리를
lexical-heavy/mixed/non-lexical로 분류 후 재평가:

| subset | n(lexical-heavy) | Global | Local | Local−Global |
|---|---|---|---|---|
| docvqa | 49 | 0.8145 | 0.7928 | −0.022 |
| arxivqa | 94 | 0.9191 | 0.8852 | −0.034 |
| gov_reports | 40 | 0.8804 | 0.8410 | −0.039 |
| infovqa | 159 | 0.8618 | 0.8145 | −0.047 |
| healthcare | 27 | 0.9630 | 0.8937 | −0.069 |
| syntheticDocQA_ai | 24 | 0.9484 | 0.8757 | −0.073 |
| energy | 21 | 0.8601 | 0.7690 | −0.091 |
| tabfquad | 161 | 0.5704 | 0.4643 | −0.106 |
| tatdqa | 526 | 0.6929 | 0.5799 | −0.113 |
| shiftproject | 30 | 0.3960 | 0.2298 | **−0.166** |

**Lexical-heavy 그룹에서도 10/10 전부 Global이 이김.** 오히려 10개 중 7개
subset에서 lexical-heavy 그룹의 격차가 전체 평균보다 더 컸다 — 가설의
정반대 방향.

## 3. RQ4 — Bounding Box Grounding (ViDoRe v3)

검색 순위와 무관하게, "Local sparse가 고른 최고점 지역이 실제 정답 근거
위치와 겹치는가"를 검증. ViDoRe v3의 bounding box 주석을 사용.

전체 bbox 주석 데이터셋(subset당 8,766~13,068개 (query,doc) 쌍, corpus
전체에 해당) 기준:

| subset | n | Region Hit@1 | Mean IoU (top1) | Mean IoU (랜덤) | 배율 |
|---|---|---|---|---|---|
| finance_en | 8,766 | 59.5% | 0.090 | 0.055 | 1.63x |
| hr | 10,368 | 72.6% | 0.104 | 0.074 | 1.40x |
| physics | 13,068 | 83.2% | 0.120 | 0.075 | 1.61x |

**3개 subset 전부에서 랜덤보다 확실히 우수.** 수식/기호가 많은 physics에서
오히려 Region Hit@1이 가장 높았다(83.2%).

> **소표본 함정 주의**: 초기에 속도를 위해 subset당 150개 샘플(전체의
> 1.5~2%)만으로 실행했을 때는 엄격한 threshold(IoU>0.3) 기준 배율이
> finance_en 3.50x, physics 4.50x로 훨씬 크게 나왔었다. 전체 데이터셋으로
> 재실행하니 1.6~1.8x대로 수렴했다 — 표본이 작을 때 엄격한 threshold를
> 넘는 관측치 자체가 적어 분산이 컸던 것. **위 표(전체 데이터셋 버전)가
> 최종 수치**이며, n=150 결과는 참고용으로만 남긴다.

### 왜 검색은 지고 grounding은 이기는가 — 수학적 설명

Sparse 벡터는 전부 0 이상(log1p(relu(·)) 변환)이다. Global 벡터는 모든
지역에 대해 차원별 max를 취한 것이므로, 정의상 특정 지역 하나의 값보다
항상 크거나 같다. 쿼리 벡터도 0 이상이므로:

```
dot(query, global_vec) ≥ dot(query, 특정 지역 하나)  (항상 성립)
따라서
dot(query, global_vec) ≥ max_k dot(query, 지역_k) = local_score  (항상 성립)
```

즉 **같은 문서에 대해 global 점수는 수학적으로 local 점수보다 절대 작을 수
없다.** Global은 "이 단어가 문서 어딘가에 있냐"를 단어별로 OR로 묶은
것이고, Local은 "이 단어들이 같은 지역 안에 다 같이 있냐"를 요구하는 AND
조건이다. 정답 문서 자신의 lexical 단서가 여러 지역에 흩어져 있는 경우가
많아서(표의 항목명·연도·숫자가 서로 다른 셀/타일에 위치), 검색(수천 개
문서와 경쟁)에서는 이 AND 조건이 정답 문서 자신에게 불리하게 작용한다.
반면 grounding은 이미 정답 문서가 정해진 뒤 그 안에서만 위치를 찾는
비경쟁 상황이라, 같은 AND 조건이 오히려 정밀함으로 작용한다.

## 4. Local-objective Fine-tuning 실험

"Global로만 학습돼서 Local이 불리했다"는 가설을 직접 검증하기 위해, 같은
베이스 체크포인트(`ModernVBERT/modernvbert`)·같은 데이터(2,000개 이미지
슬라이스)·같은 step 수(500 step)에서 **손실함수만 Global vs Local로 다르게**
학습시킨 두 LoRA를 비교(소규모 proof-of-concept, 논문 원 레시피와 조건이
다름 — 8GPU가 아닌 1GPU, 5e-5가 아닌 1e-5, FLOPS reg/caption loss 비활성화 등).

| 학습 objective | 평가 방식 | nDCG@5 |
|---|---|---|
| 모델 A: Global (원 릴리스와 동일한 손실) | global-mode | 0.4158 |
| 모델 A: Global | **local-mode** | **0.3586** |
| 모델 B: Local (신규 구현) | global-mode | 0.3971 |
| 모델 B: Local | **local-mode** | **0.3568** |

**핵심 비교**: Local로 학습한 모델을 local-mode로 평가한 것(0.3568)이,
Global로 학습한 모델을 local-mode로 평가한 것(0.3586)보다 오히려 근소하게
낮음. **"학습 목적함수 불일치"가 Local 열세의 주 원인이라는 가설은 이
실험에서 지지되지 않음.** 500 step짜리 소규모 실험이라 단정하긴 이르지만,
적어도 이 규모에서는 "Local(타일) 표현 자체가 구조적으로 정보량이 적다"는
더 근본적인 설명 쪽에 무게가 실린다.

## 5. 최종 결론

1. **Local Sparse가 Global Sparse보다 검색 성능이 낫다는 핵심 가설은
   기각됨** — 10개 subset 전부, 전체 평균과 lexical-heavy 그룹 양쪽 다.
2. **Local Sparse는 실제 근거 위치를 찾는 능력(grounding)에서는 랜덤보다
   확실히 우수함** — 검색 순위와 위치 파악력은 서로 다른 능력.
3. **원인은 "학습 부족"보다 "표현 자체의 구조적 정보량 한계"에 가까움** —
   Global의 OR 방식(단서가 문서 어디에 있든 인정)이 Local의 AND 방식(단서가
   같은 지역에 몰려있어야 인정)보다 검색(경쟁 상황)에서 수학적으로 항상
   유리하다.
4. **실무적 시사점**: 검색은 Global/Dense로 하고, 최종 후보 문서가 정해진
   뒤에만 Local sparse로 근거 위치를 짚어주는 2단계 구조로 쓰면 Local의
   약점 없이 위치 특정 능력만 얻을 수 있다.

## 한계

- BiModernVBERT dense-only 자체의 재현 성능이 논문 공식 수치(nDCG@5 ~0.68)에
  훨씬 못 미침(우리 재현 0.377) — 여러 가설(배치/정밀도/해상도/가중치로딩)을
  직접 테스트해 배제했으나 정확한 원인은 찾지 못함.
- Lexical-heavy 자동 라벨을 사람이 재검토하지 않음.
- Local fine-tuning 실험은 500 step/이미지 2,000개짜리 소규모 실험.
- bbox grounding은 ViDoRe v3 8개 subset 중 3개(finance_en, hr, physics)만 검증.
