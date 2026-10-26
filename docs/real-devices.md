# Running the pool on real devices

Several real laptops, each lending what it can, working as one. Everyone does the same
three steps. Nobody types an address, edits a file, or installs Node.

## What you need

Each person, on their own laptop:

- **Windows, macOS or Linux**, with a few GB free. A GPU (NVIDIA) makes it faster but is
  not needed.
- **Internet, the first time only.** It downloads PyTorch and the model, about 3 GB.
  After that nothing is downloaded.
- **The same Wi-Fi as the others** (or a phone hotspot one of you turns on). If the Wi-Fi
  blocks devices from seeing each other, see [Different networks](#different-networks).
- **Git.** Everything else installs itself.

## The three steps

**1. Get the code.**

```bash
git clone https://github.com/YUVARAJ-R-ai/Torrent_llm.git
cd Torrent_llm
git switch main
```

**2. Join.**

| System | Command |
|---|---|
| Linux, macOS | `scripts/join.sh` |
| Windows | `powershell -ExecutionPolicy Bypass -File scripts\join.ps1` |

The first run installs everything and downloads the model, so it takes several minutes.
It then checks the machine and starts the device. A later run starts in seconds.

**3. Open the page it prints.**

```
  Torrent-LLM pool device "alices-laptop" is running.

  Open this in a browser to see and use the pool:   http://192.168.1.20:9000
```

Open that address, on any laptop on the network. The page is served by the device
itself, so you can open *any* device's address and see the same pool.

As the others run `join.sh`, they appear on the page within a few seconds. You do
nothing.

## Check it works

Once everyone has joined, run this from any laptop (it does not need to be in the pool):

```bash
python scripts/pool_check.py http://192.168.1.20:9000      # any device's address
```

It asks every device the same question, checks they all give the same right answer, shows
the route each took and how long it took, and asks five questions at once. It prints `FAIL`
lines for anything wrong and exits non-zero. If it passes, the pool works across your
machines.

## Things to try

| Do this | What you should see |
|---|---|
| Ask a question from different laptops' pages | The same answer, taking different routes through the devices |
| Close one laptop's lid, or press Ctrl+C in its terminal | The page marks it gone; questions keep working while any copy of each stage is up |
| Close it **mid-answer** | The answer still arrives, after a short pause |
| Start a game, or anything heavy, on a laptop with an NVIDIA GPU | The device notices and hands its work to the others, then lends the GPU again when you stop |
| Press **Owner starts a game** on the page | The same, on demand |
| Add a fourth laptop | It appears and takes a share of the work |

With two laptops the page says **Degraded**: each stage is on one device only. That is
correct. It becomes **Healthy** when every stage is on two devices. A laptop that lends
enough memory holds both stages itself, which is why lending more helps.

## If something goes wrong

Run the check. It says what is wrong and how to fix it:

```bash
~/Coding/venvs/torrent-llm/bin/torrent-peer --check        # Linux, macOS
```
```powershell
& "$HOME\Coding\venvs\torrent-llm\Scripts\torrent-peer.exe" --check    # Windows
```

(`join.sh` and `join.ps1` already run it before starting, and stop if something is wrong.)

| Symptom | Cause and fix |
|---|---|
| **The others do not appear** | Almost always the network. Wi-Fi at colleges and cafés often stops devices seeing each other. Start a phone hotspot and join it, or point one device at another: `scripts/join.sh --join 192.168.1.20:9000` (use the address the first device printed). |
| **They appear, then show as offline** | A firewall is blocking the model ports. Windows: allow python on private networks when it asks, or run the commands `--check` prints in an Administrator PowerShell. Ports: TCP 9000 and 9200-9201, UDP 9099. |
| **"Model is not downloaded"** | The first setup did not finish, or you were offline. Connect and run `scripts/setup.sh` (Windows: `scripts\setup.ps1`). |
| **"Joins but holds nothing"** | Less memory is free than one stage needs. Close other programs, or `--memory-mb 3000` if you have the room. |
| **"Another device is also called ..."** | Two laptops have the same name. Give each its own: `--name alice`. |
| **"Port 9000 already in use"** | A second copy is running, or another program has the port. Stop it, or use `--port 9001`. |
| **Everything is slow** | A CPU-only laptop is much slower than a GPU one; that is expected. The page shows which device is which. |

## Different networks

Devices on different networks (one at home, one at a friend's) cannot find each other by
broadcast, and routers block incoming connections. The simple fix is
[Tailscale](https://tailscale.com/download): install it on every laptop, sign in to the
same account, then:

```bash
scripts/join.sh --host tailscale                              # the first laptop
scripts/join.sh --host tailscale --join 100.101.102.103:9000  # the others, with its Tailscale address
```

## Try compression

On a slow link the hidden states crossing between devices are the main cost. A second pool
file shrinks them 8x using a codec calibrated on real activations, at a cost of a few
points of answer quality (see [compression.md](compression.md)):

```bash
scripts/join.sh --pool configs/pool-compressed.yaml
```

**Every device must use the same pool file.** If one does not, the devices refuse each
other and the page says why. The page then shows "compressed" in its top bar and how many
times smaller the traffic was after each answer. To compare, run the same questions with
and without it.

## Keeping strangers out

By default anyone on the network can use the pool. To require a shared secret, give every
device the same one:

```bash
scripts/join.sh --token "pick-a-long-random-phrase"
```

Devices without it are refused by the other devices and by their model ports. The page's
**owner** buttons change a device, so they need the token too: open the page as
`http://192.168.1.20:9000/?token=pick-a-long-random-phrase`. Asking questions never needs it. This is
access control, not encryption: on a network you do not trust, use Tailscale as well.

## Options

`torrent-peer --help` lists everything. The ones you may want:

| Option | Meaning |
|---|---|
| `--name` | The device's name on the page. Default: the computer's name |
| `--memory-mb` | How much to lend. Default: a share of what is free |
| `--device` | `cpu`, `cuda` or `mps`. Default: the best available |
| `--join host:port` | Connect to a device directly instead of searching |
| `--host` | The address others use: `auto`, `lan`, `tailscale`, or an address |
| `--no-discover` | Do not broadcast; use `--join` only |
| `--no-watch-owner` | Do not back off when other programs use the GPU |
| `--port` | The web port. Default 9000 |
| `--token` | Shared secret |

## What has and has not been tested

Tested: two real device processes finding each other with no address, serving the page on
the LAN address, answering through either, on Linux with an NVIDIA GPU; all the logic with
automated tests; the network pieces in-process over real sockets.

**Not tested yet:** Windows and macOS (the Windows script has not been run at all, only
written to mirror the tested setup script), three or more separate machines, a real
router, Tailscale, and automatic owner detection on Windows (NVIDIA's tool reports less
there, so it may simply never trigger; the button still works). If something is off on
your machine, the check and the terminal output are what to send.
