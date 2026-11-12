S="${S:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # needs $S/ssh_config with hosts A-D, F; results go to $S/results
TOKEN=$(openssl rand -hex 16)
r() { ssh -n -F "$S/ssh_config" "$@"; }
declare -A IP=([A]=172.31.32.228 [B]=172.31.43.255 [C]=172.31.38.84 [D]=172.31.39.44 [F]=47.129.187.240 [Apub]=13.234.37.234)
PEER=/home/ubuntu/Coding/venvs/torrent-llm/bin/torrent-peer
PY=/home/ubuntu/Coding/venvs/torrent-llm/bin/python
stop_peer() { r $1 'pkill -9 -f "[t]orrent-peer"; pkill -9 -f "[t]orrent_llm"; true'; }
# start_peer HOST POOLFILE ADVERTISED_IP MEMORY_MB [JOIN_ADDR]
start_peer() { r $1 "cd ~/Torrent_llm || exit 1; HF_HUB_OFFLINE=1 nohup $PEER --pool $2 --name $1 --device cpu --host $3 --memory-mb $4 --no-discover --no-watch-owner --token $TOKEN ${5:+--join $5} > ~/peer.log 2>&1 < /dev/null & echo started $1"; }
nb() { local h=$1; shift; r $h "$PY ~/netbench.py $*"; }
# shape HOST PEER_IP RATE DELAY_MS : netem on HOST's egress to PEER_IP only (ssh untouched)
shape() { r $1 "sudo tc qdisc del dev enp39s0 root 2>/dev/null; sudo tc qdisc add dev enp39s0 root handle 1: prio bands 4 priomap 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 && sudo tc qdisc add dev enp39s0 parent 1:4 handle 40: netem delay ${4}ms rate $3 limit 100000 && sudo tc filter add dev enp39s0 parent 1: protocol ip prio 1 u32 match ip dst $2/32 flowid 1:4"; }
unshape() { r $1 'sudo tc qdisc del dev enp39s0 root 2>/dev/null; true'; }
