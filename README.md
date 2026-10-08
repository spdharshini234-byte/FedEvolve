# Federated Continual Learning for Privacy-Preserving Visual Defect Detection

**Federated Continual Learning Framework for Privacy-Preserving Visual Defect Detection under Non-IID Distributions and Evolving Defect Classes**

A software-only academic project. **The factories are simulated software clients** running on one computer.

---
## 1. Project overview
Three simulated factories train a ResNet-18 defect classifier together. Each factory keeps its images locally and sends only model weights to a central server, which combines them with **FedAvg**. Later a **new defect class** appears and the model learns it with **replay (rehearsal)** while trying not to forget the old classes. A FastAPI backend and a React website show the real results.

## 2. Problem statement
1. Factories do not want to share product images (privacy / business secrets).
2. Each factory sees different defects (**non-IID** data).
3. New defect types appear over time. A model fine-tuned only on the new defect forgets the old ones (**catastrophic forgetting**).

## 3. Proposed solution
Federated Learning (FedAvg) for collaboration without sharing images + Continual Learning with a small local replay buffer for new classes.

## 4. System architecture
```
 Factory A      Factory B      Factory C        (local images stay here; replay buffer is local too)
    |  local training  |               |
    +---- weights only (no images) ----+
                       v
              Central server -> FedAvg -> Global model
                       |
        global model sent back to every factory  (repeat for N rounds)

 React (Vite)  <--HTTP/JSON-->  FastAPI  <-->  src/ (PyTorch pipeline)  <-->  results/, models/
```

## 5. Technologies
Python 3.11/3.12, PyTorch + torchvision (ResNet-18), scikit-learn, NumPy, Pandas, Matplotlib, Pillow, PyYAML, FastAPI, React + Vite. CPU only; no cloud, no paid API.

## 6. Algorithms
* **Transfer learning**: ResNet-18 (ImageNet weights if they can be downloaded; otherwise random initialisation with a warning) with a new final layer.
* **FedAvg**: `global = sum(n_k / n * w_k)` (`src/federated/fedavg.py`), implemented by hand.
* **Replay**: class-balanced buffer of old local samples (`src/continual/replay_buffer.py`).
* **Forgetting score**: average drop of accuracy on an old task (best earlier accuracy minus final accuracy).

## 7. Dataset format
```
data/raw/
    scratch/  *.jpg|png
    crack/
    dent/
    color_defect/
```
Class names come from `configs/config.yaml` (`dataset.classes`). See `data/README.md`. **No real dataset is included.** `python run.py prepare-demo` creates a *synthetic* demo dataset (drawn images) so everything can run.

## 8. Installation
Python 3.11 or 3.12 and Node.js 18+ are needed.

## 9. Windows PowerShell setup
```powershell
cd federated-continual-defect-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd frontend; npm install; cd ..
```
Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`, then the same `pip` commands.

## 10. How to prepare data
Own images: put them in `data/raw/<class>/`, then
```powershell
python run.py verify-data
python run.py prepare-data      # resize + stratified train/val/test split
```
No dataset: `python run.py prepare-demo` (synthetic images + split + clients in one step).

## 11. How to create clients
```powershell
python run.py create-clients    # non-IID split of the TRAIN images into 3 factories
python scripts/create_continual_tasks.py   # optional: writes the task plan
```
Non-IID behaviour is set in `configs/federated_config.yaml` (`non_iid`): per class, the share of its images given to each factory (`matrix` mode) or a Dirichlet split.
Default: Factory A mostly scratch + some crack; B mostly crack + some dent; C mostly dent/color defect + some scratch.
Continual tasks are in `configs/continual_config.yaml` (default: Task 1 = scratch, crack; Task 2 = dent; Task 3 = color_defect).

## 12. How to run experiments
```powershell
python run.py train-centralized --demo
python run.py train-local --demo
python run.py train-fedavg --demo
python run.py train-continual --demo   # runs sequential FedAvg WITHOUT replay (shows forgetting) and WITH replay
python run.py evaluate                 # re-evaluates saved checkpoints, rebuilds results/experiment_results.csv and plots
python run.py full-experiment --demo   # everything above
```
`--demo` only means "prepare the synthetic demo data if it is not prepared yet". Outputs: `results/metrics/*.json`, `results/experiment_results.csv`, `results/confusion_matrices/`, `results/plots/`, `models/global/*.pt`, `models/client/*.pt`.
Settings (rounds, epochs, batch size, image size, learning rate, replay buffer size, number of clients) are in `configs/*.yaml`.

## 13. How to start FastAPI
From the project root (venv active):
```powershell
uvicorn backend.app.main:app --port 8000
```
API docs: http://127.0.0.1:8000/docs

## 14. How to start React
```powershell
cd frontend
npm run dev
```
Open http://localhost:5173 (the dev server forwards `/api` to port 8000).

## 15. How to use the website
Dashboard (summary) · Factory Clients (images per factory, class distribution) · Federated Training (buttons start real runs; live round/client status; history) · Continual Learning (old/new class accuracy, replay buffer, forgetting) · Model Comparison · Prediction (upload an image) · Confusion Matrix · Privacy / Communication.

## 16. Expected workflow
1. `prepare-demo` (or your own data + `prepare-data` + `create-clients`).
2. Run the experiments (CLI or website buttons).
3. Look at Model Comparison, Continual Learning and Confusion Matrix.
4. Upload an image on the Prediction page.

## 17. Evaluation metrics
Accuracy, precision, recall, F1, macro-F1, confusion matrix, per-class metrics, old-class and new-class accuracy, forgetting score, communication cost. Everything is evaluated on the same held-out test split. "Old classes" = classes of all tasks before the last one; "new class" = the last task's classes (the same definition is used for all experiments so they are comparable). In *Local Only*, each factory model can only predict classes that factory has; metrics pool the predictions of the three local models.

## 18. Federated Learning explanation
Each round: the server sends the global weights; every factory trains for a few local epochs on its own images; every factory sends back its weights; the server averages them (weighted by number of local images). `Server.receive_update` accepts only a dict of tensors, so images cannot be uploaded through the project's API. Communication cost = bytes of the model weights x (downloads + uploads) x rounds, measured from the real model.

## 19. Continual Learning explanation
When a new task arrives, each factory trains on its new-class images **plus** its replay buffer (a few remembered old images, kept locally). Without replay the model quickly forgets old classes. The *Continual Learning* page compares both runs, using real numbers from the saved results. During a task the output layer is masked to the classes seen so far.

## 20. Privacy limitations
* Raw training images remain local in the federated setup, but **basic Federated Learning does not guarantee complete privacy**. Model updates can leak information (e.g. gradient/membership inference).
* **Secure aggregation and differential privacy can be added as future improvements.**
* The server in this project is trusted and there is no authentication or encryption.

## 21. Project limitations
* The factories are simulated software clients on one machine; there is no real network.
* No real dataset is included. The demo images are synthetic and easy; results on them are not evidence for real factories. Use a real dataset for any claim.
* The pipeline is simple (small images, few epochs) so it runs on a laptop CPU; accuracy is modest and run-to-run noise is large on tiny test sets.
* Default 64x64 inputs and short training. ImageNet weights need internet access the first time; without them the code warns and trains from scratch.
* This is not claimed to be novel, industrial-ready, or secure.

## 22. Future improvements
Secure aggregation, differential privacy, FedProx / other non-IID methods, better continual methods (distillation, EWC), more clients, real network simulation, larger images and real datasets, experiment tracking.

## Tests
```powershell
pytest -q
```
Covers dataset loading, model creation, FedAvg, replay buffer, continual metrics, API endpoints and a smoke test of the full pipeline (tiny settings).

## Project layout
See the folder tree: `configs/`, `data/`, `scripts/`, `src/` (data, models, training, federated, continual, evaluation, utils, `pipeline.py`), `backend/`, `frontend/`, `notebooks/`, `tests/`, `results/`, `models/`.
