#!/usr/bin/env bash
# =====================================================================
#  DLoc Lightweight Localization — one-shot environment setup
# =====================================================================
#  Reproduces a working project folder from the three published sources:
#    * code             -> GitHub  (this repo)
#    * dataset (~70 GB) -> HF  JustAGeek/dloc-wild-fig10b   (dataset)
#    * trained weights  -> HF  JustAGeek/dloc-code          (model)
#    * project memory   -> HF  JustAGeek/dloc-private       (PRIVATE, needs access)
#
#  Resulting layout (everything runs from  code/ ):
#    DLoc-lightweight-localization/
#      code/                 source (core, models, training, evaluation, configs)
#      code/data/            the WILD .mat dataset               (from HF)
#      code/weights/         10 trained .pth checkpoints         (from HF)
#      claude-memory/        20 Claude memory .md files          (from HF, private)
#      data_split_idx/ Figures/ reports/ docs/ thesis/ ...       (in the git clone)
#
#  USAGE
#    bash setup.sh                 # code + venv + weights + memory + symlinks,
#                                  #   and PROMPT before the ~70 GB dataset
#    bash setup.sh --all -y        # everything, no prompts (incl. dataset)
#    bash setup.sh --no-dataset    # skip the big dataset download
#    bash setup.sh --with-teacher  # also regenerate teacher outputs (needs a GPU)
#    bash setup.sh --help
#
#  PREREQUISITES: git, python3 (>=3.9), pip, internet.
#  (huggingface_hub is installed into the venv by this script.)
# =====================================================================
set -euo pipefail

# ---- configuration ---------------------------------------------------
REPO_URL="https://github.com/Abdulrahman-bits2/DLoc-lightweight-localization.git"
REPO_DIR="DLoc-lightweight-localization"
HF_DATASET="JustAGeek/dloc-wild-fig10b"   # dataset repo (the .mat files)
HF_WEIGHTS="JustAGeek/dloc-code"          # model repo   (weights/, logs/, results/)
HF_PRIVATE="JustAGeek/dloc-private"       # PRIVATE repo (claude-memory/)

# ---- options ---------------------------------------------------------
WANT_DATASET="prompt"; WANT_WEIGHTS=1; WANT_MEMORY=1; WANT_TEACHER=0; ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --all)          WANT_DATASET=1; WANT_WEIGHTS=1; WANT_MEMORY=1 ;;
    --no-dataset)   WANT_DATASET=0 ;;
    --with-dataset) WANT_DATASET=1 ;;
    --no-weights)   WANT_WEIGHTS=0 ;;
    --no-memory)    WANT_MEMORY=0 ;;
    --with-teacher) WANT_TEACHER=1 ;;
    -y|--yes)       ASSUME_YES=1 ;;
    -h|--help)      sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg  (try --help)"; exit 2 ;;
  esac
done

# ---- helpers ---------------------------------------------------------
c_ok=$'\033[32m'; c_hd=$'\033[1;36m'; c_wn=$'\033[33m'; c_x=$'\033[0m'
step() { echo; echo "${c_hd}==> $*${c_x}"; }
ok()   { echo "${c_ok}  [ok] $*${c_x}"; }
warn() { echo "${c_wn}  [!] $*${c_x}"; }
die()  { echo "  [ERROR] $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
confirm() { [ "$ASSUME_YES" = 1 ] && return 0; read -r -p "  $1 [y/N] " a; [[ "$a" =~ ^[Yy]$ ]]; }

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) VENV_BIN=".venv/Scripts" ;;
  *)                    VENV_BIN=".venv/bin"     ;;
esac

# download a HF repo using whichever CLI the venv provides (hf is newer)
hf_dl() { # hf_dl <repo> <repo-type> <local-dir> [extra args...]
  local repo="$1" rtype="$2" ldir="$3"; shift 3
  if have hf; then
    hf download "$repo" --repo-type "$rtype" --local-dir "$ldir" "$@"
  else
    huggingface-cli download "$repo" --repo-type "$rtype" --local-dir "$ldir" "$@"
  fi
}

# =====================================================================
step "0/7  Checking prerequisites"
have git || die "git not found — install git first."
PY="$(command -v python3 || command -v python || true)"; [ -n "$PY" ] || die "python3 not found (need >=3.9)."
echo "  git    : $(git --version)"
echo "  python : $("$PY" --version 2>&1)"
ok "prerequisites present"

# =====================================================================
step "1/7  Cloning the code repository"
if [ -d "$REPO_DIR/.git" ]; then
  ok "repo already present ($REPO_DIR) — pulling latest"
  ( cd "$REPO_DIR" && git pull --ff-only || warn "could not fast-forward; leaving as-is" )
elif [ -d "code" ] && [ -f "REPRODUCIBILITY.md" ]; then
  REPO_DIR="."; ok "already inside a clone — using current directory"
else
  git clone "$REPO_URL" "$REPO_DIR"; ok "cloned into $REPO_DIR"
fi
cd "$REPO_DIR"; ROOT="$(pwd)"

# =====================================================================
step "2/7  Python environment + dependencies"
[ -d ".venv" ] && ok ".venv exists" || { "$PY" -m venv .venv; ok "created .venv"; }
export PATH="$ROOT/$VENV_BIN:$PATH"          # venv python/pip/hf take precedence from here on
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r code/requirements.txt
have hf || have huggingface-cli || python -m pip install --quiet "huggingface_hub[cli]"
ok "dependencies installed (code/requirements.txt)"
warn "Mamba student only: also  'pip install mamba-ssm --no-deps && pip install causal-conv1d'  (CUDA-only)"

# =====================================================================
step "3/7  Dataset  ($HF_DATASET  ~70 GB)  ->  code/data/"
[ "$WANT_DATASET" = "prompt" ] && { confirm "Download the ~70 GB WILD dataset now?" && WANT_DATASET=1 || WANT_DATASET=0; }
if [ "$WANT_DATASET" = 1 ]; then
  ( cd code && hf_dl "$HF_DATASET" dataset data ) || die "dataset download failed — re-run (hf resumes)."
  ok "dataset in code/data/"
else
  warn "skipped dataset — later:  cd code && hf download $HF_DATASET --repo-type dataset --local-dir data"
fi

# =====================================================================
step "4/7  Trained weights  ($HF_WEIGHTS)  ->  code/weights/"
if [ "$WANT_WEIGHTS" = 1 ]; then
  ( cd code && hf_dl "$HF_WEIGHTS" model . --include "weights/*" ) || die "weights download failed — re-run."
  ok "10 checkpoints in code/weights/"
else
  warn "skipped weights"
fi

# =====================================================================
step "5/7  Project memory  ($HF_PRIVATE, PRIVATE)  ->  claude-memory/"
if [ "$WANT_MEMORY" = 1 ]; then
  if python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
    if hf_dl "$HF_PRIVATE" dataset . ; then
      ok "memory in claude-memory/ (open your Claude account here; tell it to read claude-memory/MEMORY.md)"
    else
      warn "memory download failed — ask the owner to add you as a collaborator on $HF_PRIVATE"
    fi
  else
    warn "not logged in to Hugging Face. Run  'hf auth login'  (you must be a collaborator on the"
    warn "private repo), then:  hf download $HF_PRIVATE --repo-type dataset --local-dir ."
  fi
else
  warn "skipped memory"
fi

# =====================================================================
step "6/7  Cross-session symlinks  (LOSO needs the dataset_train_ names)"
if ls code/data/dataset_jacobs_Aug16_1.mat >/dev/null 2>&1; then
  ( cd code/data
    for s in 1 3 4_ref; do
      dst="dataset_train_jacobs_Aug16_${s}.mat"
      [ -e "$dst" ] || ln -s "dataset_jacobs_Aug16_${s}.mat" "$dst" 2>/dev/null \
        || warn "could not symlink $dst (Windows: enable Developer Mode, or use 'mklink')"
    done )
  ok "Aug16 train_ symlinks ready (only needed for the LOSO cross-session runs)"
else
  warn "dataset not present — symlinks skipped (re-run after the dataset download)"
fi

# =====================================================================
step "7/7  Teacher outputs  (distillation runs only; needs a GPU + dataset + baseline weights)"
if [ "$WANT_TEACHER" = 1 ]; then
  ( cd code && python training/precompute_teacher.py ) && ok "teacher_*.pt generated" \
    || warn "teacher precompute failed (needs GPU + code/data + baseline weights)"
else
  warn "skipped — only needed for DISTILLED runs. Later:  cd code && python training/precompute_teacher.py"
fi

# =====================================================================
echo; echo "${c_hd}=====================================================================${c_x}"
echo "${c_ok} Setup complete.  Project root: $ROOT${c_x}"
echo "${c_hd}=====================================================================${c_x}"
cat <<EOF

 Next steps (run everything from the code/ folder):
   cd "$ROOT/code"
   source "$ROOT/$VENV_BIN/activate"        # Windows cmd: $ROOT\\$VENV_BIN\\activate

   # Verify published results WITHOUT a GPU (uses the downloaded weights):
   python evaluation/benchmark_models.py                 # efficiency table (no data needed)
   python evaluation/test_robustness.py                  # set MODEL_TYPE + WEIGHTS_PATH at top
   python evaluation/eval_cross_session.py               # set MODEL_TYPE + WEIGHTS_PATH at top

   # Retrain (needs a GPU):
   python training/precompute_teacher.py                 # once, before any distilled run
   python training/train_distillation.py                 # students (edit STUDENT_TYPE / MODE)
   python training/train_unet_advanced.py                # U-Net (edit SCHEDULER_TYPE / MODE)
   bash    training/run_crosssession_table1.sh           # LOSO 3 folds x 3 seeds

 Reference:  REPRODUCIBILITY.md (full result->command map) · ONBOARDING.md ·
             Experiments_and_Data_Splits.{tex,pdf} (every split + run documented).
 Note: WILD_PASSWORD is only needed to RE-download raw CSI via the kaggle_*.ipynb
       notebooks; normal use pulls the processed data from Hugging Face above.
EOF
