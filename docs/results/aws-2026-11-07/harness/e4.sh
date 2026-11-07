#!/bin/bash
# Experiment 4: answer quality on the 95 held-out questions through a real 2-device pool (C <-> D).
S="${S:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # needs $S/ssh_config with hosts A-D, F; results go to $S/results
source $S/drv.sh
out=$S/results/e4-quality.jsonl; : > $out
for pool in pool pool-compressed pool-c16; do
  start_peer D configs/$pool.yaml ${IP[D]} 1600
  sleep 2
  start_peer C configs/$pool.yaml ${IP[C]} 1600 ${IP[D]}:9000
  nb C wait http://127.0.0.1:9000 --members 2 --copies 1 --timeout 200 >/dev/null || echo "WAIT FAILED $pool"
  nb C quality http://127.0.0.1:9000 --label $pool | tee -a $out | cut -c1-200
  stop_peer C; stop_peer D; sleep 2
done
echo E4 DONE
