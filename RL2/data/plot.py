import re, os, glob
from datetime import timedelta
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates

# ------------------------------------------------------------
#  CONFIG
# ------------------------------------------------------------
TZ_SHIFT = timedelta(hours=4)               # shift log → UTC metrics
LOG_FILE = "log.txt"
metrics_files = sorted(glob.glob("metrics_*.jsonl"))
if not metrics_files:
    print("❌ No metrics_*.jsonl files found in current directory")
    print("Available files:", os.listdir("."))
    exit(1)
METRICS = metrics_files[-1]  # newest
print(f"📊 Using metrics file: {METRICS}")
OUT_PDF  = "all_main_tests.pdf"

# ------------------------------------------------------------
#  LOAD METRICS  (epoch-seconds ➜ datetime)
# ------------------------------------------------------------
raw = pd.read_json(StringIO(open(METRICS).read()), lines=True)
raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="s")

# --- expand switch_mlus safely ---
if "switch_mlus" in raw.columns:
    sw = raw["switch_mlus"].apply(pd.Series)
    sw.columns = [f"switch_{c}" for c in sw.columns]
    # force every cell to scalar numeric
    for col in sw.columns:
        sw[col] = pd.to_numeric(sw[col], errors="coerce")
    raw = pd.concat([raw.drop(columns=["switch_mlus"]), sw], axis=1)

# ------------------------------------------------------------
#  PARSE log.txt  ➜ main windows & TCP spans
# ------------------------------------------------------------
pat_main_start = re.compile(r"\[(.*?)\].*MAIN TEST ITERATION (\d+) STARTED")
pat_main_end   = re.compile(r"\[(.*?)\].*Main iteration (\d+) completed")
pat_tcp_start  = re.compile(r"\[(.*?)\].*\[TCP-Competitor\] Starting TCP test")
pat_tcp_end    = re.compile(r"\[(.*?)\].*\[TCP-Competitor\] TCP test .* completed")

main = {}
tcp  = {}

with open(LOG_FILE) as f:
    for ln in f:
        if m := pat_main_start.search(ln):
            ts  = pd.to_datetime(m.group(1)) + TZ_SHIFT
            idx = int(m.group(2))
            main[idx] = [ts, None]
            tcp[idx]  = []
        elif m := pat_main_end.search(ln):
            ts  = pd.to_datetime(m.group(1)) + TZ_SHIFT
            idx = int(m.group(2))
            if idx in main: main[idx][1] = ts
        elif m := pat_tcp_start.search(ln):
            ts = pd.to_datetime(m.group(1)) + TZ_SHIFT
            for idx,(s,e) in main.items():
                if s and e and s <= ts <= e:
                    tcp[idx].append([ts, None]); break
        elif m := pat_tcp_end.search(ln):
            ts = pd.to_datetime(m.group(1)) + TZ_SHIFT
            for spans in tcp.values():
                if spans and spans[-1][1] is None:
                    spans[-1][1] = ts
                    break

windows = [(i, *main[i]) for i in sorted(main) if all(main[i])]

# ------------------------------------------------------------
#  HELPER
# ------------------------------------------------------------
def clip_flow(s, lim=15):
    good = s[s<=lim]
    mean = good.mean() if not good.empty else 0
    return s.where(s<=lim, mean), mean

# ------------------------------------------------------------
#  PLOT (four stacked panels)  +  SHADED TCP AREAS
# ------------------------------------------------------------
palette = ["red","green","blue","purple"]
date_fmt = mdates.DateFormatter("%m-%d\n%H:%M")

with PdfPages(OUT_PDF) as pdf:
    for idx, start, end in windows:
        df = raw[(raw["timestamp"]>=start) & (raw["timestamp"]<=end)]
        if df.empty:
            continue

        flow, mean_f = clip_flow(df["aggregated_flow_rate_mbps"])
        sw_cols = [c for c in df.columns if c.startswith("switch_")]

        fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        fig.suptitle(f"MAIN TEST {idx}", fontsize=15, fontweight="bold")

        # ---- SHADE competing-TCP spans ----
        for s,e in tcp.get(idx, []):
            if not e: continue
            for a in ax:
                a.axvspan(s, e, color="lightgray", alpha=0.7, zorder=0)

        # Panel 1 – STD
        ax[0].plot(df["timestamp"], df["sdlus_stddev"], color="red",  label="Switch STD")
        ax[0].plot(df["timestamp"], df["link_stddev"],  color="green",label="Link STD")
        ax[0].set_ylabel("STD"); ax[0].legend()

        # Panel 2 – Max-MLU
        ax[1].plot(df["timestamp"], df["max_mlu"], color="red")
        ax[1].set_ylabel("MLU")

        # Panel 3 – Flow w/ mean line
        ax[2].plot(df["timestamp"], flow, color="red")
        ax[2].axhline(mean_f, linestyle="--", color="red")
        ax[2].set_ylabel("Mbps")

        # Panel 4 – per-switch MLUs
        for i, col in enumerate(sw_cols):
            ax[3].plot(df["timestamp"], df[col],
                       color=palette[i%4], label=str(i+1))
        ax[3].set_ylabel("MLU"); ax[3].legend(title="SW")

        # single time axis
        ax[-1].set_xlabel("Time")
        ax[-1].xaxis.set_major_formatter(date_fmt)

        plt.tight_layout(rect=[0,0.04,1,0.96])
        pdf.savefig(fig)
        plt.close(fig)

print("✅  Finished.  PDF written to", os.path.abspath(OUT_PDF))
