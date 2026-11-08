"""Run the local models in a process that the API can restart."""
import sys
from pathlib import Path

from .main import Analysis, read, write


def run(video, job, launch_video=None, hive_weights=None):
    def progress(amount, stage):
        write(job / "progress.json", [amount, stage])

    try:
        progress(.01, "Loading the model runtime")
        from .vision import analyze

        config = Analysis.model_validate(read(job / "config.json"))
        write(job / "result.json", analyze(video, config, progress, launch_video, hive_weights))
    except Exception as exc:
        write(job / "error.json", str(exc)[:240] or "Analysis failed.")
        raise


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
