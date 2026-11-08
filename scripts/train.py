"""Fine-tune the lightweight FTC detector from YOLO-format annotations."""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="YOLO dataset YAML with match-disjoint train and val splits")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--name", default="ftc")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/yolo26n.pt")
    parser.add_argument("--tight-crops", action="store_true", help="Keep object scale and layout close to a marked Flower crop")
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        parser.error("--name must contain only letters, numbers, hyphens or underscores")
    model = YOLO(args.weights)
    augmentation = {"mosaic": 0, "scale": .18, "translate": .06} if args.tight_crops else {}
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.size,
                batch=args.batch, workers=0, cache=False, device=args.device, seed=408,
                project=str(ROOT / "runs"), name=args.name, plots=False, **augmentation)
    destination = ROOT / "models" / f"{args.name}.pt"
    shutil.copyfile(model.trainer.best, destination)
    print(f"Checkpoint ready: {destination}")


if __name__ == "__main__":
    main()
