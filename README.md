# SAC-VRS (Variance-Regularized Soft Actor-Critic via Reward Scaling)

## 프로젝트 소개 & 주요 성과
[KIPS ACK 2025에서 발표한 **Variance-Regularized Soft Actor-Critic (SAC-VRS)**](https://koreascience.kr/article/CFKO202532457662089.page) 구현 리포지터리입니다. OpenAI SpinningUp의 SAC/PPO 코드를 Windows 친화적으로 리팩터링하고, **Reward Scaling + Variance Regularization**을 적용해 BipedalWalker-v3, Ant-v4와 같은 박스2D·MuJoCo 환경에서 학습 안정성을 높였습니다.
- `sac/` 모듈은 표준 SAC 학습 루프에 **스케일링 보상, Q-variance 모니터링, Windows용 로거**를 통합했습니다.
- `ppo/optuna_tune.py`는 Optuna 기반 하이퍼파라미터 탐색 파이프라인으로 PPO baseline을 튜닝합니다.
- `AntTest_SAC.ipynb`, `bipedalTest_SAC*.ipynb`는 Colab/로컬에서 Reward Scaling 전후 지표를 비교하고 논문 Figure를 재현합니다.
- `data/` 폴더에는 trial별 로그, `Accepted Paper & Presentation/`에는 ACK 2025 논문/발표 자료가 포함되어 있습니다.

---

## 프로젝트 배경
- **BipedalWalker-v3**와 **Ant-v4**는 고차원 연속 제어 문제로, 보상 분포가 비대칭/고분산이어서 Q 함수 학습이 불안정합니다.
- 기존 SAC는 **고정 온도 파라미터**나 **리플레이 버퍼 샘플링**만으로는 Q1/Q2 편차를 제어하기 어렵고, 결과적으로 정책이 과감한 행동을 학습하지 못합니다.
- 본 프로젝트는 **Reward Scaling & Variance Regularization**을 추가하여, 학습 초반의 극단적 보상을 완화하고 Q-gap을 줄이는 방식을 실험했습니다.

---

## 기술 스택
- **언어/프레임워크**: Python 3.10+, PyTorch 2.x, Gymnasium(Box2D, MuJoCo), NumPy, Optuna
- **실험/시각화**: Matplotlib, pandas, tqdm, Colab Notebooks
- **로깅/유틸**: `utils/logx_single.py` (Windows용 Logger), `utils/run_utils.py`, Joblib, Cloudpickle
- **배포/문서**: ACK 2025 Paper & Presentation (PDF)

---

## 환경 & 데이터셋 (시뮬레이터)
| 환경 | 설명 | 공식 링크 |
|------|------|-----------|
| **BipedalWalker-v3** (Box2D) | 24-D 관측 / 4-D 연속 제어, 보상 -150~300 | https://gymnasium.farama.org/environments/box2d/bipedal_walker/ |
| **Ant-v4** (MuJoCo) | 111-D 관측 / 8-D 연속 제어, 보상 0~600 | https://gymnasium.farama.org/environments/mujoco/ant/ |

- Box2D 기반 환경은 `pip install gymnasium[box2d]`로 바로 실행 가능하며, MuJoCo 환경은 `mujoco` 패키지와 mjkey 라이선스(또는 v3 오픈소스)를 설치해야 합니다.
- 각 실험의 리플레이/로그는 `data/` 폴더에 저장됩니다. (예: `data/sac_optimal/sac_optimal_s0/progress.txt`)

---

## 모델 & 알고리즘 하이라이트
1. **Reward Scaling**  
   - 에피소드별 원시 보상을 표준화(`μ`, `σ`)한 뒤 Q-target 계산 시 스케일링하여 학습 초반 variance를 낮췄습니다.
2. **Variance Regularization**  
   - `AverageQ1Vals`, `AverageQ2Vals`, `AverageQTarget`, `Q1-Q2 gap`을 매 epoch 기록하고, 편차가 커질 경우 scaling coefficient를 자동 조정합니다.
3. **Adaptive Logging**  
   - `utils/logx_single.py`로 MPI 없이도 SpinningUp 스타일 로그를 출력/저장할 수 있습니다.
4. **Optuna Search (PPO)**  
   - `ppo/optuna_tune.py`로 `gamma`, `hidden size`, `steps_per_epoch` 등을 자동 탐색하여 baseline 대비 향상폭을 정량화했습니다.

---

## 실험 워크플로
1. **환경 준비**  
   ```bash
   conda create -n sac-vrs python=3.11 -y
   conda activate sac-vrs
   pip install torch gymnasium[box2d] mujoco optuna matplotlib pandas tqdm
   ```
2. **SAC 학습**  
   ```bash
   cd SAC-VRS
   python -m sac.sac --env BipedalWalker-v3 --hid 256 --l 2 --epochs 200 --exp_name sac_baseline
   python -m sac.sac --env BipedalWalker-v3 --hid 256 --l 2 --epochs 200 --exp_name sac_vrs --alpha 0.2
   ```
   - Reward Scaling 파라미터는 `AntTest_SAC.ipynb` 내 함수(`apply_reward_scaling`)를 참고하거나 커스텀 래퍼를 추가합니다.
3. **PPO + Optuna**  
   ```bash
   cd SAC-VRS/ppo
   python optuna_tune.py  # 내부에서 ppo_main.py 호출 (경로 수정 필요)
   ```
4. **노트북 재현**  
   - `bipedalTest_SAC.ipynb`: Colab/로컬에서 baseline vs. scaled 로그(`progress.txt`)를 로드하고 그래프 출력.
   - `AntTest_SAC.ipynb`: Ant-v4 실험, reward scaling 적용 후 평균 보상 및 Q-variance 그래프 생성.
5. **결과 확인**  
   - `data/<exp_name>/<exp_name>_s0/progress.txt`에서 Epoch별 지표 확인.
   - `Accepted Paper & Presentation/`에서 논문/발표 자료 확인.

---

## 디렉터리 안내
```
SAC-VRS/
├── sac/                     # SAC 알고리즘 (core.py, sac.py)
├── ppo/                     # PPO baseline + Optuna 튜닝 스크립트
├── utils/                   # Windows 호환 로거 & 런처
├── data/                    # trial, sac_optimal 로그 및 모델
├── AntTest_SAC.ipynb        # Ant-v4 실험 재현
├── bipedalTest_SAC*.ipynb   # BipedalWalker 실험 노트북 (baseline/변형)
├── Accepted Paper & Presentation/ # 발표된 논문과 논문 발표 자료
└── LICENSE / README.md
```

---

## 핵심 결과
- **BipedalWalker-v3**  
  - Baseline SAC: `AverageTestEpRet ≈ -110` 수준에서 진동, Q-gap > 15.  
  - SAC-VRS: 30 epoch 이후 `AverageTestEpRet ≈ -2 ~ -5`로 수렴, `StdQ1Vals`가 7.9→6.7로 감소.
- **Ant-v4**  
  - Reward Scaling 적용 시 `AverageEpRet` 상승 속도가 15% 빨라지고, 시험 환경에서 `StdTestEpRet`이 40% 감소.
- **Optuna PPO**  
  - 20 trials 중 best trial은 `gamma=0.99`, `hid=256`, `steps=4000` 조합에서 baseline 대비 +18% reward를 기록했습니다.

---

## 사용 팁 & 참고 사항
- **Windows 환경**: `utils/logx_single.py`, `utils/run_utils.py`는 MPI 의존성을 제거하므로 `mpiexec` 없이 실행 가능합니다.
- **로그 시각화**: `pandas.read_csv('progress.txt', sep='\t')` 후 Matplotlib으로 그래프를 그리거나 Notebooks를 활용하세요.
- **모델 저장**: `data/<exp>/pyt_save/model.pt`로 저장되며, `torch.load` 후 `ac.act()`를 통해 정책을 재사용할 수 있습니다.

---

## 향후 계획
1. Reward Scaling 계수를 자동으로 학습하는 메타-컨트롤러 도입.
2. Ant-v4, Walker 등 MuJoCo 환경에서 PPO+SAC 하이브리드 실험 추가.
3. Logged 데이터로 **Offline RL** (CQL, IQL) 실험 확장.
4. 논문/발표 자료에 쓰인 그래프를 `matplotlib` 스크립트로 분리하여 CLI에서도 재현 가능하도록 공개.

---

## 개인 기여
- SAC 코드 베이스를 Windows/Colab에서도 동작하도록 리팩터링 (`utils/logx_single`, MPI 제거).
- Reward Scaling, Q-variance 모니터링 로직 구현 및 실험 지표 시각화.
- Optuna 기반 PPO 하이퍼파라미터 탐색 파이프라인 구축.
- ACK 2025 논문 및 발표 자료 작성, BipedalWalker/Ant 실험 결과 보고.

---

## 참고 자료
- KIPS ACK 2025 논문 & 발표: `Accepted Paper & Presentation/SAC-VRS_*.pdf`
- OpenAI SpinningUp (SAC/PPO reference): https://spinningup.openai.com/
- Gymnasium Environments: https://gymnasium.farama.org/
- Optuna Hyperparameter Optimization: https://optuna.org/
