#!/usr/bin/env python3
"""debug_hang - watch a job's output; on a hang, dump backtraces, release the job.

A hang is silence: the output log stops growing for --timeout seconds. When that
happens, debug_hang collects a rocgdb backtrace from every application rank on
every node of the job, writes them to a directory, and cancels the job so the
nodes return to the pool.

Two ways to run it:

  wrap a command:   debug_hang.py --mpi "srun -N2 -n8" --timeout 300 -- ./app        (Slurm)
                    debug_hang.py --mpi "mpirun -n 8 --ppn 4" --timeout 300 -- ./app  (PBS)
  watch a log:      debug_hang.py --timeout 300 --log job.out

--mpi works exactly like the launcher scripts under scripts/: pass the MPI launch
command as data, the trailing command after -- is just the application. Omit
--mpi to run/watch a single rank with no MPI at all. The scheduler (Slurm or
PBS) is auto-detected from the job environment, falling back to a local,
single-node/no-scheduler mode when neither is present; load your ROCm/therock
module before launching and debug_hang carries that environment to the collect
step.

The `collect` subcommand runs on each node and is invoked by debug_hang itself.
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime


HELP_BLURB = """\
Watches your MPI job's output for a hang -- no output for a set number of
seconds -- and when that happens, automatically grabs a snapshot of every
rank's call stack (on every node of the job) so you can see exactly where it
got stuck, without knowing anything about attaching a debugger yourself. The
snapshot is written to files on disk, then the job is cancelled so the
allocation isn't wasted (unless you pass --no-cancel).

This only detects silence, not an immediate crash: a rank that segfaults
outright just exits, which counts as "no hang" here -- use debug_crash.sh
instead if your job is dying outright rather than hanging. Auto-detects
Slurm or PBS from the job environment (or runs standalone, single-rank, no
scheduler needed); load your ROCm module before launching so the snapshot
step debugs against the same ROCm build your application used.

Under the hood, this uses AMD's rocgdb -- see
https://rocm.docs.amd.com/projects/ROCgdb/en/latest/ for details.
"""


def log(msg):
    print(f"[debug_hang {datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


# --- scheduler ---------------------------------------------------------------

def detect_scheduler():
    if os.environ.get("SLURM_JOB_ID"):
        return "slurm"
    if os.environ.get("PBS_JOBID"):
        return "pbs"
    if shutil.which("scancel"):
        return "slurm"
    if shutil.which("qdel"):
        return "pbs"
    return "local"


def check_dependencies(scheduler):
    """Fail fast and clearly, before any watching/dispatch starts.

    Only the fan-out tool the detected scheduler actually needs is required --
    a local/no-scheduler single-rank run must not be blocked on pbsdsh/srun
    being installed, since dispatch() never calls either of them in that case.
    """
    if not shutil.which("rocgdb"):
        sys.exit("error: 'rocgdb' not found on PATH. Load the ROCm module providing it and retry.")
    if scheduler == "pbs" and not shutil.which("pbsdsh"):
        sys.exit("error: 'pbsdsh' not found on PATH (needed to fan out across PBS nodes).")
    if scheduler == "slurm" and not shutil.which("srun"):
        sys.exit("error: 'srun' not found on PATH (needed to fan out across Slurm nodes).")


def job_id(scheduler):
    if scheduler == "slurm":
        jid = os.environ.get("SLURM_JOB_ID")
        if jid:
            return jid
        out = run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%A"])
        return out.split("\n")[0].strip() if out else None
    if scheduler == "pbs":
        return os.environ.get("PBS_JOBID")
    return None


def nodes(scheduler, jid):
    if scheduler == "slurm":
        spec = os.environ.get("SLURM_JOB_NODELIST")
        if not spec and jid:
            spec = run(["squeue", "-h", "-j", jid, "-o", "%N"]).strip()
        if spec:
            return run(["scontrol", "show", "hostnames", spec]).split()
    if scheduler == "pbs":
        nf = os.environ.get("PBS_NODEFILE")
        if nf and os.path.exists(nf):
            seen = []
            for line in open(nf):
                n = line.strip()
                if n and n not in seen:
                    seen.append(n)
            return seen
    return [os.uname().nodename]


def pbs_node_indices(node_list):
    """Map each unique host to its first line index in PBS_NODEFILE.

    pbsdsh -n addresses nodes by their position in the nodefile, which may
    list a host on several lines (one per chunk/CPU). The deduped node_list
    order does not track those positions, so look the indices up directly.
    """
    nf = os.environ.get("PBS_NODEFILE")
    first = {}
    if nf and os.path.exists(nf):
        for i, line in enumerate(open(nf)):
            n = line.strip()
            if n and n not in first:
                first[n] = i
    # Fall back to positional index for any host not found in the nodefile.
    return [first.get(node, idx) for idx, node in enumerate(node_list)]


def run(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL,
                              universal_newlines=True).stdout
    except FileNotFoundError:
        return ""


# --- collection (runs on each node) ------------------------------------------

def gpu_pids():
    """PIDs holding a GPU context (the application ranks), owned by this user."""
    out = run(["rocm-smi", "--showpids"])
    pids = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\b", line)
        if m:
            pids.append(int(m.group(1)))
    return [p for p in pids if owned_by_me(p)]


def pattern_pids(pattern):
    out = run(["pgrep", "-u", os.environ["USER"], "-f", pattern])
    return [int(p) for p in out.split()]


def owned_by_me(pid):
    try:
        st = os.stat(f"/proc/{pid}")
        return st.st_uid == os.getuid()
    except OSError:
        return False


def proc_name(pid):
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return "?"


def backtrace(pid, path, timeout, env=None):
    cmd = [
        "rocgdb", "-p", str(pid), "--batch", "-nx",
        "-ex", "set pagination off",
        "-ex", "thread apply all bt",
    ]
    # `env` is debug_hang's own environment (captured after the user's modules
    # were loaded) carried to this collect task. Running rocgdb under it pins
    # amd-dbgapi to the SAME rocgdb/runtime the app used, so GPU debugging
    # works. Without it the pbsdsh task inherits the job's default environment
    # and rocgdb fails with an r_debug version mismatch (host stacks only).
    with open(path, "w") as f:
        f.write(f"# pid {pid} ({proc_name(pid)}) on {os.uname().nodename}\n\n")
        f.flush()
        try:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            f.write(f"\n# rocgdb timed out after {timeout}s\n")


def load_env(path):
    """Read a NUL-separated env dump (from `printenv0`) into a dict."""
    if not path or not os.path.exists(path):
        return None
    env = {}
    with open(path, "rb") as f:
        for entry in f.read().split(b"\0"):
            if b"=" in entry:
                k, v = entry.split(b"=", 1)
                env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return env or None


def collect(args):
    host = os.uname().nodename
    outdir = os.path.join(args.outdir, host)
    os.makedirs(outdir, exist_ok=True)
    env = load_env(args.envfile)
    pids = pattern_pids(args.pattern) if args.pattern else gpu_pids()
    log(f"{host}: {len(pids)} target pid(s): {pids}")
    for pid in pids:
        backtrace(pid, os.path.join(outdir, f"pid{pid}.bt"),
                  args.rocgdb_timeout, env)
    with open(os.path.join(outdir, "manifest.txt"), "w") as f:
        for pid in pids:
            f.write(f"{pid}\t{proc_name(pid)}\n")
    return 0


# --- fan out over the job's nodes --------------------------------------------

def gpus_on(node):
    """GPUs per node, so rocgdb on the collect step can read wavefronts."""
    m = re.search(r"gpu:(\d+)", run(["sinfo", "-N", "-h", "-n", node, "-o", "%G"]))
    return int(m.group(1)) if m else 0


def dump_env(outdir):
    """Snapshot debug_hang's own environment (post module-load) to a file on the
    shared FS, so the collect step can run rocgdb with the exact PATH /
    LD_LIBRARY_PATH the app used. NUL-separated to survive values with newlines.
    """
    path = os.path.join(outdir, "debug_hang.env")
    with open(path, "wb") as f:
        for k, v in os.environ.items():
            f.write(f"{k}={v}".encode("utf-8", "replace") + b"\0")
    return path


def dispatch(args, scheduler, jid, node_list):
    self_path = os.path.abspath(__file__)
    envfile = dump_env(args.outdir)
    inner = [sys.executable, self_path, "collect", "--outdir", args.outdir,
             "--rocgdb-timeout", str(args.rocgdb_timeout), "--envfile", envfile]
    if args.pattern:
        inner += ["--pattern", args.pattern]

    pbs_idx = pbs_node_indices(node_list) if scheduler == "pbs" else None

    def remote(index, node):
        if scheduler == "slurm":
            gpn = args.gpus_per_node if args.gpus_per_node >= 0 else gpus_on(node)
            cmd = ["srun", "--jobid", jid, "--overlap", "-N1", "-n1", "-c4", "-w", node]
            if gpn > 0:
                cmd.append(f"--gpus-per-node={gpn}")
            cmd += inner
        elif scheduler == "pbs":
            # pbsdsh addresses nodes by their index in the job allocation,
            # not by hostname; -n runs one copy on the index-th node.
            cmd = ["pbsdsh", "-n", str(pbs_idx[index]), "--"] + inner
        else:
            cmd = inner
        subprocess.run(cmd)

    threads = [threading.Thread(target=remote, args=(i, n))
               for i, n in enumerate(node_list)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def cancel(scheduler, jid):
    if scheduler == "slurm":
        subprocess.run(["scancel", jid])
    elif scheduler == "pbs":
        subprocess.run(["qdel", jid])
    else:
        log("no scheduler: nothing to cancel")


# --- monitor -----------------------------------------------------------------

def tee(stream, path):
    """Copy the child's output to both the watched log and our stdout.

    Read line by line, not in fixed-size blocks: a buffered read(N) blocks
    until N bytes accumulate, so low-rate output (e.g. a heartbeat every few
    seconds) would never reach the log and debug_hang would see false silence.
    Flush after every line so the watched file grows in real time.
    """
    with open(path, "wb", buffering=0) as f:
        for line in iter(stream.readline, b""):
            f.write(line)
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()


def size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def watch(args):
    scheduler = args.scheduler if args.scheduler != "auto" else detect_scheduler()
    jid = job_id(scheduler)
    node_list = nodes(scheduler, jid)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.outdir = os.path.abspath(args.outdir or f"debug_hang-{jid or 'local'}-{stamp}")
    mpi_prefix = shlex.split(args.mpi) if args.mpi else []
    full_command = mpi_prefix + args.command if args.command else []

    log(f"scheduler={scheduler} job={jid} nodes={node_list}")
    log(f"watching {args.log} (timeout {args.timeout}s) -> {args.outdir}")

    check_dependencies(scheduler)

    if args.dry_run:
        if full_command:
            log(f"would launch: {' '.join(full_command)}")
        log(f"would collect backtraces to: {args.outdir}")
        return 0

    proc = None
    if full_command:
        # bufsize=0 keeps our read side unbuffered so low-rate output reaches
        # the watched log immediately. The child (and any mpirun/srun launcher
        # between us and it) may still fully buffer stdout because it is a pipe,
        # not a TTY; the app should fflush() its own output, or be run under a
        # line-buffering shim (e.g. `stdbuf -oL -eL` / `unbuffer`).
        proc = subprocess.Popen(full_command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=0)
        log(f"launched: {' '.join(full_command)} (pid {proc.pid})")
        threading.Thread(target=tee, args=(proc.stdout, args.log),
                         daemon=True).start()

    last_size = -1
    last_change = time.time()
    while True:
        if proc is not None and proc.poll() is not None:
            log(f"command exited ({proc.returncode}); no hang")
            return proc.returncode
        cur = size(args.log)
        now = time.time()
        if cur != last_size:
            last_size, last_change = cur, now
        elif cur >= 0 and now - last_change >= args.timeout:
            break
        time.sleep(args.poll)

    log(f"HANG: no output for {args.timeout}s -> collecting backtraces")
    os.makedirs(args.outdir, exist_ok=True)
    dispatch(args, scheduler, jid, node_list)
    log(f"backtraces in {args.outdir}")
    if args.no_cancel or jid is None:
        log("leaving job running (--no-cancel)")
    else:
        log(f"canceling job {jid}")
        cancel(scheduler, jid)
    return 0


# --- cli ---------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(prog="debug_hang.py", description=HELP_BLURB,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timeout", type=int, default=300,
                   help="seconds of log silence that counts as a hang (default 300)")
    p.add_argument("--log", help="log file to watch (created in wrap mode)")
    p.add_argument("--outdir", help="where backtraces go (default debug_hang-<job>-<ts>)")
    p.add_argument("--mpi", help='MPI launch command to prefix the command with '
                        '(e.g. "mpirun -np 4"); omit for a single rank, no MPI')
    p.add_argument("--pattern", help="regex for target process; default is GPU ranks")
    p.add_argument("--poll", type=float, default=5.0, help="poll interval seconds")
    p.add_argument("--rocgdb-timeout", type=int, default=60, help="per-rank rocgdb budget")
    p.add_argument("--gpus-per-node", type=int, default=-1,
                   help="Slurm only: GPUs for the collect step (-1 auto-detects "
                        "via sinfo, 0 disables); ignored under PBS")
    p.add_argument("--scheduler", choices=["auto", "slurm", "pbs", "local"],
                   default="auto", help="scheduler to use (auto-detects from "
                        "SLURM_JOB_ID / PBS_JOBID)")
    p.add_argument("--no-cancel", action="store_true", help="collect but do not cancel")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would run, don't watch/launch/collect")
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="command to run after -- (wrap mode)")
    return p


def parse_args(argv=None):
    """Parse and normalize CLI args: strip the leading `--` off `command` (if
    present), require a command or a --log, and default --log from the PID
    when only a command was given. Split out from main() so it's testable
    without triggering watch()'s real subprocess/monitoring side effects.
    """
    argv = sys.argv[1:] if argv is None else argv
    p = build_arg_parser()
    args = p.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command and not args.log:
        p.error("give a command (-- CMD ...) or a --log to watch")
    if args.command and not args.log:
        args.log = f"debug_hang-{os.getpid()}.log"
    return args


def main():
    # `collect` runs on each node; route it before the main parser so the
    # `-- CMD ...` remainder of watch mode stays clean.
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        c = argparse.ArgumentParser(prog="debug_hang.py collect")
        c.add_argument("--outdir", required=True)
        c.add_argument("--pattern")
        c.add_argument("--rocgdb-timeout", type=int, default=60)
        c.add_argument("--envfile")
        return collect(c.parse_args(sys.argv[2:]))

    return watch(parse_args())


if __name__ == "__main__":
    sys.exit(main())
