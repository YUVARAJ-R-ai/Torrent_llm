#!/bin/bash
# Experiment 3: real WAN, A (Mumbai) <-> F (Singapore), public addresses, no shaping.
S="${S:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # needs $S/ssh_config with hosts A-D, F; results go to $S/results
source $S/drv.sh
out=$S/results/e3-wan.jsonl; : > $out
for pool in pool pool-compressed pool-c16; do
  start_peer F configs/$pool.yaml ${IP[F]} 1600
  sleep 2
  start_peer A configs/$pool.yaml ${IP[Apub]} 1600 ${IP[F]}:9000
  nb A wait http://127.0.0.1:9000 --members 2 --copies 1 --timeout 200 >/dev/null || echo "WAIT FAILED $pool"
  r A 'curl -s localhost:9000/pool' | python3 -c 'import json,sys; v=json.load(sys.stdin); print([(m["name"],m["host"],m["hosting"],m.get("rtt_ms")) for m in v["members"]])'
  res=$(nb A speed http://127.0.0.1:9000 --label $pool --repeats 5 --tokens 16)
  echo "{\"link\":\"wan-mumbai-singapore\",\"result\":$res}" | tee -a $out
  stop_peer A; stop_peer F; sleep 2
done
echo E3 DONE
