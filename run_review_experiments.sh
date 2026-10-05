#!/bin/zsh
# Runs the review experiments E1-E11 one after another (low priority, 1 BLAS thread).
# Start (from anywhere):  nohup zsh export/run_review_experiments.sh > export/results/review/runner.log 2>&1 &
# Check progress:         zsh export/check_progress.sh
cd "${0:A:h}"
OUT=results/review
mkdir -p $OUT/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=. PYTHONUNBUFFERED=1 WORKERS=${WORKERS:-2}
PY=${PY:-python3}
PYHF=${PYHF:-$HOME/miniconda3/envs/dmwithllm/bin/python}
STATUS=$OUT/status.txt
# Networks that inspect HTTPS need the Mac's trusted certificates for the model downloads in E2.
if [[ -z $REQUESTS_CA_BUNDLE ]] && security find-certificate -a -p /Library/Keychains/System.keychain \
     /System/Library/Keychains/SystemRootCertificates.keychain > $OUT/mac-ca.pem 2>/dev/null; then
  export REQUESTS_CA_BUNDLE=$PWD/$OUT/mac-ca.pem SSL_CERT_FILE=$PWD/$OUT/mac-ca.pem
fi
echo "$(date '+%F %T') RUNNER START (pid $$, WORKERS=$WORKERS)" > $STATUS

run() {
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> $STATUS
  nice -n 10 "$@" > $OUT/logs/$name.log 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name exit=$rc" >> $STATUS
}

run E6_lemma_checks     $PY   verification/lemma_fix_check.py
run E1_value_matrices   $PY   verification/value_matrix_sweep.py
run E2_ov_sign          $PYHF verification/ov_sign_check.py
run E3_triangle_sweep   $PY   verification/triangle_sweep_large.py
run E4_integrator       $PY   verification/integrator_check.py
run E5_highdim          $PY   verification/highdim_check.py
run E8_lowrank_value    $PY   verification/lowrank_value_check.py
run E8b_lowrank_ext    env EXT=1 $PY verification/lowrank_value_check.py
run E3b_triangle_longtime $PY verification/triangle_longtime.py
run E9_unequal_triangle $PY   verification/unequal_triangle_check.py
run E7_trained_heads    $PYHF verification/trained_head_flow.py
run E10_antisym_share   $PY   verification/antisym_share.py
run E11_antisym_depth   env PYTHONPATH=.:verification $PYHF verification/antisym_depth.py
echo "$(date '+%F %T') ALL DONE" >> $STATUS
