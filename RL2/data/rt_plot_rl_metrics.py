"""
live_metrics_plot.py  watch a JSON-Lines metrics file and plot in real time
⚠️  Run in a normal Python session (not inside Ryu), e.g.  python live_metrics_plot.py
"""

import json, time, pathlib, matplotlib.pyplot as plt
import matplotlib.animation as animation
import typing as _t
from pathlib import Path

def _get_latest_metrics_file(metrics_dir: _t.Union[str, Path]) -> Path:
    """
    Return the Path of the newest *.jsonl file inside `metrics_dir`.
    Raises FileNotFoundError if none exist.

    Example
    -------
    latest_file = _get_latest_metrics_file(Path(__file__).parent / "Metrics")
    """
    metrics_dir = Path(metrics_dir)
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"{metrics_dir} does not exist or is not a directory")

    jsonl_files = list(metrics_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No *.jsonl files found in {metrics_dir}")

    # max() with key=os.path.getmtime would also work; Path.stat() is slightly cleaner
    newest = max(jsonl_files, key=lambda p: p.stat().st_mtime)
    return newest
# ---------------------- Configure ------------------------------------------------
FILE = _get_latest_metrics_file("RL2/data/Metrics")  # adjust pattern / path
METRIC_KEY = "aggregate_rate_mbps"        # what to plot (any numeric field)
REFRESH_MS = 500                          # animation interval
# -------------------------------------------------------------------------------

def tail_jsonl(fp, read_existing=True, sleep_sec=0.3):
    """
    Yield parsed JSON objects from a growing *.jsonl file.

    Parameters
    ----------
    fp : open file object
    read_existing : bool
        • True  ➜ stream the whole file from the top, then follow new lines  
        • False ➜ skip current contents (classic `tail -f`)
    sleep_sec : float
        Polling sleep while waiting for new data.
    """
    if not read_existing:
        fp.seek(0, 2)                       # jump to EOF

    while True:
        line = fp.readline()
        if not line:
            time.sleep(sleep_sec)
            continue

        # Some editors write partial lines; ignore until '\n' seen
        if not line.endswith("\n"):
            # Seek back so the partial chunk will be read again later
            fp.seek(-len(line), 1)
            time.sleep(sleep_sec)
            continue

        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue                        # malformed line; skip
                    # skip malformed partial lines

# ---------------------- Plot set-up --------------------------------------------
x_data, y_data = [], []
fig, ax = plt.subplots()
(line_plot,) = ax.plot([], [], lw=2)
ax.set_xlabel("Time (s, epoch)")
ax.set_ylabel(METRIC_KEY)
ax.set_title(f"Live {METRIC_KEY} from {FILE.name}")

def on_new_record(rec):
    """Update buffers and line object for each incoming record."""
    x_data.append(rec["timestamp"])
    y_data.append(rec.get(METRIC_KEY, 0))
    line_plot.set_data(x_data, y_data)
    ax.relim()
    ax.autoscale_view()
    return line_plot,

def data_stream():
    """Generator that feeds FuncAnimation."""
    with open(FILE, "r") as fp:
        for rec in tail_jsonl(fp):
            yield rec

ani = animation.FuncAnimation(
    fig,
    func=lambda rec: on_new_record(rec),
    frames=data_stream,        # <- your generator
    interval=REFRESH_MS,
    blit=False,
    cache_frame_data=False     # <-- add this (or save_count=1000, etc.)
)

plt.show()
