#!/usr/bin/env bash
# Tüm hattı sıfırdan yeniden üretir. Ham Excel dosyaları data/raw/ altında olmalı.
# Kullanım: ./run_all.sh            (RAG dahil, GPU önerilir)
#           SKIP_RAG=1 ./run_all.sh (yalnız veri + regresyon, CPU, ~2 dk)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
[ -x .venv/bin/python ] && PY="$(pwd)/.venv/bin/python"
cd src
echo "== 1 veri hattı";      $PY s01_ingest.py && $PY s02_clean.py && $PY s03_match.py && $PY s04_eda.py && $PY s05_splits.py
echo "== 2 matematik";       $PY math_checks.py
echo "== 3 eğitim";          $PY train.py main && $PY train.py ablations && $PY train.py leakage
echo "== 4 değerlendirme";   $PY evaluate.py && $PY error_analysis.py && $PY posthoc_smearing.py
echo "== 5 altın sorular";   (cd rag && $PY build_gold.py)
if [ "${SKIP_RAG:-0}" != "1" ]; then
  echo "== 6 RAG"; (cd rag && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY eval_rag.py dev \
                    && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY eval_rag.py test && $PY eval_rag.py manual)
fi
cd ..
echo "== 7 testler";         $PY -m pytest tests -q
