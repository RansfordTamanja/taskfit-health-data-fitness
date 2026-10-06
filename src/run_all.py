"""run_all.py: full TaskFit pipeline (about 45 minutes on one CPU core)."""
import subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent
steps = [["run_experiments.py"], ["-c", "import analysis, analysis2; analysis.main(); analysis.extended(); analysis.final_tests(); analysis2.main()"], ["figures.py"]]
for s in steps:
    print("\n===", " ".join(s), flush=True)
    if subprocess.run([sys.executable, "-W", "ignore", *s], cwd=SRC).returncode != 0:
        sys.exit(f"FAILED at {s}")
print("Pipeline complete. See outputs/tables and outputs/figures.")
