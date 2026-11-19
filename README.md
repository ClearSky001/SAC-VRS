# SAC-VRS (Variance-Regularized Soft Actor-Critic via Reward Scaling)

**English** | [한국어](#한국어)

## Project Overview & Key Results
This repository implements [**Variance-Regularized Soft Actor-Critic (SAC-VRS)**, presented at KIPS ACK 2025](https://koreascience.kr/article/CFKO202532457662089.page). We refactored the SAC/PPO code from OpenAI SpinningUp to be Windows-friendly and applied **Reward Scaling + Variance Regularization** to improve training stability in Box2D and MuJoCo environments such as BipedalWalker-v3 and Ant-v4.
- The `sac/` module integrates **scaled rewards, Q-variance monitoring, and a Windows logger** into the standard SAC training loop.
- `ppo/optuna_tune.py` is an Optuna-based hyperparameter search pipeline that tunes the PPO baseline.
- `AntTest_SAC.ipynb` and `bipedalTest_SAC*.ipynb` compare metrics before and after Reward Scaling in Colab or locally, and reproduce the paper's figures.
- The `data/` folder holds per-trial logs, and `Accepted Paper & Presentation/` contains the ACK 2025 paper and presentation materials.

---

## Background
- **BipedalWalker-v3** and **Ant-v4** are high-dimensional continuous control problems with skewed, high-variance reward distributions, which makes Q-function learning unstable.
- With only a **fixed temperature parameter** or **replay buffer sampling**, standard SAC struggles to control the gap between Q1 and Q2, and as a result the policy fails to learn bold actions.
- This project added **Reward Scaling & Variance Regularization** and tested an approach that softens extreme rewards early in training and reduces the Q-gap.

---

## Tech Stack
- **Languages/Frameworks**: Python 3.10+, PyTorch 2.x, Gymnasium (Box2D, MuJoCo), NumPy, Optuna
- **Experiments/Visualization**: Matplotlib, pandas, tqdm, Colab Notebooks
- **Logging/Utilities**: `utils/logx_single.py` (Windows logger), `utils/run_utils.py`, Joblib, Cloudpickle
- **Publication/Docs**: ACK 2025 Paper & Presentation (PDF)

---

## Environments & Datasets (Simulators)
| Environment | Description | Official link |
|------|------|-----------|
| **BipedalWalker-v3** (Box2D) | 24-D observations / 4-D continuous control, reward -150~300 | https://gymnasium.farama.org/environments/box2d/bipedal_walker/ |
| **Ant-v4** (MuJoCo) | 111-D observations / 8-D continuous control, reward 0~600 | https://gymnasium.farama.org/environments/mujoco/ant/ |

- Box2D environments run right away after `pip install gymnasium[box2d]`; MuJoCo environments require the `mujoco` package and an mjkey license (or the open-source v3).
- Each experiment's replays and logs are saved in the `data/` folder. (e.g., `data/sac_optimal/sac_optimal_s0/progress.txt`)

---

## Model & Algorithm Highlights
1. **Reward Scaling**  
   - Standardizes the raw per-episode rewards (`μ`, `σ`) and scales them when computing the Q-target, lowering variance early in training.
2. **Variance Regularization**  
   - Records `AverageQ1Vals`, `AverageQ2Vals`, `AverageQTarget`, and the `Q1-Q2 gap` every epoch, and automatically adjusts the scaling coefficient when the deviation grows.
3. **Adaptive Logging**  
   - `utils/logx_single.py` prints and saves SpinningUp-style logs even without MPI.
4. **Optuna Search (PPO)**  
   - `ppo/optuna_tune.py` automatically searches `gamma`, `hidden size`, `steps_per_epoch`, and more, quantifying the improvement over the baseline.

---

## Experiment Workflow
1. **Set up the environment**  
   ```bash
   conda create -n sac-vrs python=3.11 -y
   conda activate sac-vrs
   pip install torch gymnasium[box2d] mujoco optuna matplotlib pandas tqdm
   ```
2. **Train SAC**  
   ```bash
   cd SAC-VRS
   python -m sac.sac --env BipedalWalker-v3 --hid 256 --l 2 --epochs 200 --exp_name sac_baseline
   python -m sac.sac --env BipedalWalker-v3 --hid 256 --l 2 --epochs 200 --exp_name sac_vrs --alpha 0.2
   ```
   - For the Reward Scaling parameters, refer to the function in `AntTest_SAC.ipynb` (`apply_reward_scaling`) or add a custom wrapper.
3. **PPO + Optuna**  
   ```bash
   cd SAC-VRS/ppo
   python optuna_tune.py  # calls ppo_main.py internally (update the path first)
   ```
4. **Reproduce with the notebooks**  
   - `bipedalTest_SAC.ipynb`: Loads the baseline vs. scaled logs (`progress.txt`) in Colab or locally and plots the graphs.
   - `AntTest_SAC.ipynb`: Ant-v4 experiment; plots the average reward and Q-variance after applying reward scaling.
5. **Check the results**  
   - See the per-epoch metrics in `data/<exp_name>/<exp_name>_s0/progress.txt`.
   - See the paper and presentation materials in `Accepted Paper & Presentation/`.

---

## Directory Guide
```
SAC-VRS/
├── sac/                     # SAC algorithm (core.py, sac.py)
├── ppo/                     # PPO baseline + Optuna tuning script
├── utils/                   # Windows-compatible logger & launcher
├── data/                    # trial and sac_optimal logs and models
├── AntTest_SAC.ipynb        # Reproduces the Ant-v4 experiment
├── bipedalTest_SAC*.ipynb   # BipedalWalker experiment notebooks (baseline/variants)
├── Accepted Paper & Presentation/ # Published paper and its presentation materials
└── LICENSE / README.md
```

---

## Key Results
- **BipedalWalker-v3**  
  - Baseline SAC: oscillates around `AverageTestEpRet ≈ -110`, with a Q-gap > 15.  
  - SAC-VRS: converges to `AverageTestEpRet ≈ -2 ~ -5` after epoch 30, and `StdQ1Vals` drops from 7.9 to 6.7.
- **Ant-v4**  
  - With Reward Scaling, `AverageEpRet` rises 15% faster, and `StdTestEpRet` in the test environment drops by 40%.
- **Optuna PPO**  
  - Out of 20 trials, the best trial (`gamma=0.99`, `hid=256`, `steps=4000`) recorded +18% reward over the baseline.

---

## Usage Tips & Notes
- **Windows**: `utils/logx_single.py` and `utils/run_utils.py` remove the MPI dependency, so you can run without `mpiexec`.
- **Log visualization**: Load the log with `pandas.read_csv('progress.txt', sep='\t')` and plot it with Matplotlib, or use the notebooks.
- **Saving models**: Models are saved to `data/<exp>/pyt_save/model.pt`; after `torch.load`, you can reuse the policy through `ac.act()`.

---

## Future Plans
1. Introduce a meta-controller that learns the Reward Scaling coefficient automatically.
2. Add PPO+SAC hybrid experiments in MuJoCo environments such as Ant-v4 and Walker.
3. Extend to **Offline RL** (CQL, IQL) experiments using the logged data.
4. Move the graphs used in the paper and presentation into `matplotlib` scripts so they can also be reproduced from the CLI.

---

## Personal Contributions
- Refactored the SAC code base to also run on Windows/Colab (`utils/logx_single`, MPI removed).
- Implemented the Reward Scaling and Q-variance monitoring logic and visualized the experiment metrics.
- Built the Optuna-based PPO hyperparameter search pipeline.
- Proposed the title of the [ACK 2025 paper](https://koreascience.kr/article/CFKO202532457662089.page), prepared the presentation materials, and reported the BipedalWalker/Ant results.

---

## References
- KIPS ACK 2025 paper & presentation: `Accepted Paper & Presentation/SAC-VRS_*.pdf`
- OpenAI SpinningUp (SAC/PPO reference): https://spinningup.openai.com/
- Gymnasium Environments: https://gymnasium.farama.org/
- Optuna Hyperparameter Optimization: https://optuna.org/

---

# 한국어

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
- [ACK 2025 논문](https://koreascience.kr/article/CFKO202532457662089.page) 제목 제안 및 발표 자료 작성, BipedalWalker/Ant 실험 결과 보고.

---

## 참고 자료
- KIPS ACK 2025 논문 & 발표: `Accepted Paper & Presentation/SAC-VRS_*.pdf`
- OpenAI SpinningUp (SAC/PPO reference): https://spinningup.openai.com/
- Gymnasium Environments: https://gymnasium.farama.org/
- Optuna Hyperparameter Optimization: https://optuna.org/
