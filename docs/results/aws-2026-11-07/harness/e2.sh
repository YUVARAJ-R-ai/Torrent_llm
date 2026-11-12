#!/bin/bash
# Experiment 2: compression vs real (netem-shaped) link, two devices A <-> B, one stage each.
S="${S:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # needs $S/ssh_config with hosts A-D, F; results go to $S/results
source $S/drv.sh
out=$S/results/e2-links.jsonl; : > $out
# rate  rtt_ms
LINKS="none:0 100mbit:10 20mbit:40 10mbit:100 5mbit:100 2mbit:100"
# reference: one device holding both stages, no network
start_peer A configs/pool.yaml ${IP[A]} 4000
nb A wait http://127.0.0.1:9000 --members 1 --copies 1 --timeout 200 >/dev/null
echo "{\"link\":\"single-device\",\"rtt_ms\":0,\"result\":$(nb A speed http://127.0.0.1:9000 --label single-device --repeats 5 --tokens 16)}" | tee -a $out
stop_peer A
for pool in pool pool-compressed pool-c16; do
  start_peer B configs/$pool.yaml ${IP[B]} 1600
  sleep 2
  start_peer A configs/$pool.yaml ${IP[A]} 1600 ${IP[B]}:9000
  nb A wait http://127.0.0.1:9000 --members 2 --copies 1 --timeout 200 >/dev/null || echo "WAIT FAILED $pool"
  for l in $LINKS; do
    rate=${l%%:*}; rtt=${l##*:}
    if [ $rate = none ]; then unshape A; unshape B
    else shape A ${IP[B]} $rate $((rtt/2)); shape B ${IP[A]} $rate $((rtt/2)); fi
    res=$(nb A speed http://127.0.0.1:9000 --label $pool --repeats 5 --tokens 16)
    echo "{\"link\":\"$rate\",\"rtt_ms\":$rtt,\"result\":$res}" | tee -a $out
  done
  unshape A; unshape B; stop_peer A; stop_peer B; sleep 2
done
echo E2 DONE
