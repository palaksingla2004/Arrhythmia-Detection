from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.core.config import ensure_runtime_dirs, settings
from app.data.unified_builder import build_unified_signal_dataset, load_unified_signal_dataset
from app.training.train_classical import train_classical_models
from app.training.train_deep import train_deep_ensemble

# Force unbuffered output
sys.stdout.flush()
sys.stderr.flush()


def run_training(
    include_ptbxl: bool = True,
    force_rebuild_dataset: bool = False,
    deep_epochs: int = 30,
    deep_batch_size: int = 256,
    quick_mode: bool = False,
    tiny_mode: bool = False,
    skip_deep: bool = False,
    skip_classical: bool = False,
) -> dict[str, Any]:
    print("=" * 80)
    print("🚀 Starting ECG Arrhythmia Training Pipeline")
    print("=" * 80)
    sys.stdout.flush()
    
    if skip_deep and skip_classical:
        raise ValueError("Both skip_deep and skip_classical are true. Nothing to train.")

    print("📁 Ensuring runtime directories...")
    sys.stdout.flush()
    ensure_runtime_dirs()
    
    dataset_npz = settings.processed_dir / "unified_signals.npz"
    print(f"📊 Dataset path: {dataset_npz}")
    print(f"🔄 Force rebuild: {force_rebuild_dataset}")
    print(f"📦 Dataset exists: {dataset_npz.exists()}")
    sys.stdout.flush()
    
    if force_rebuild_dataset or not dataset_npz.exists():
        print("\n" + "=" * 80)
        print("🔨 REBUILDING DATASET (This may take 5-30 minutes...)")
        print("=" * 80)
        sys.stdout.flush()
        
        if tiny_mode:
            print("⚡ Using TINY mode (ultra-light dataset)")
            sys.stdout.flush()
            build_unified_signal_dataset(
                include_ptbxl=False,
                mitbih_raw_max_samples=2_000,
                mitbih_train_max_rows=1_000,
                mitbih_test_max_rows=400,
                ptbdb_normal_max_rows=400,
                ptbdb_abnormal_max_rows=400,
                ptbxl_max_records=0,
            )
        elif quick_mode:
            print("⚡ Using QUICK mode (reduced dataset)")
            sys.stdout.flush()
            build_unified_signal_dataset(
                include_ptbxl=False,
                mitbih_raw_max_samples=10_000,
                mitbih_train_max_rows=5_000,
                mitbih_test_max_rows=2_000,
                ptbdb_normal_max_rows=2_000,
                ptbdb_abnormal_max_rows=2_000,
                ptbxl_max_records=0,
            )
        else:
            print("🔥 Using FULL mode (complete dataset)")
            sys.stdout.flush()
            build_unified_signal_dataset(include_ptbxl=include_ptbxl)
        print("✅ Dataset rebuild complete!")
        sys.stdout.flush()
    else:
        print("✅ Using existing dataset (no rebuild needed)")
        sys.stdout.flush()
    
    print("\n📥 Loading unified signal dataset...")
    sys.stdout.flush()
    dataset = load_unified_signal_dataset(dataset_npz)
    print(f"✅ Dataset loaded: {dataset.X.shape[0]} samples, {dataset.X.shape[1]} features")
    sys.stdout.flush()

    deep_dir = settings.models_dir / "deep"
    classical_dir = settings.models_dir / "classical"

    if skip_deep:
        print("\n⏭️  Skipping deep learning models")
        sys.stdout.flush()
        deep_artifacts = []
    else:
        print("\n" + "=" * 80)
        print(f"🧠 TRAINING DEEP LEARNING MODELS (Epochs: {deep_epochs}, Batch: {deep_batch_size})")
        print("=" * 80)
        sys.stdout.flush()
        deep_artifacts = train_deep_ensemble(
            dataset=dataset,
            output_dir=deep_dir,
            epochs=deep_epochs,
            batch_size=deep_batch_size,
        )
        print("✅ Deep learning training complete!")
        sys.stdout.flush()

    if skip_classical:
        print("\n⏭️  Skipping classical models")
        sys.stdout.flush()
        classical_summary = None
    else:
        print("\n" + "=" * 80)
        print("🎯 TRAINING CLASSICAL MODELS")
        print("=" * 80)
        sys.stdout.flush()
        classical_summary = train_classical_models(
            output_dir=classical_dir,
            sample_per_dataset=1_000 if tiny_mode else (3000 if quick_mode else 120_000),
        )
        print("✅ Classical model training complete!")
        sys.stdout.flush()

    result = {
        "deep_models": [
            {
                "model_name": a.model_name,
                "checkpoint_path": str(a.checkpoint_path),
                "calibration_path": str(a.calibration_path),
                "metrics": a.metrics,
            }
            for a in deep_artifacts
        ],
        "classical_summary": None
        if classical_summary is None
        else {
            "num_samples": classical_summary.num_samples,
            "num_features": classical_summary.num_features,
            "model_rankings": classical_summary.model_rankings,
        },
    }
    
    print("\n💾 Saving training summary...")
    sys.stdout.flush()
    (settings.models_dir / "training_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    sys.stdout.flush()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ECG arrhythmia detection stack.")
    parser.add_argument("--no-ptbxl", action="store_true", help="Skip PTB-XL during unified signal build.")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Force rebuilding the unified dataset.")
    parser.add_argument("--epochs", type=int, default=30, help="Deep model epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Deep model batch size.")
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Use smaller sample sizes for smoke-training and debugging.",
    )
    parser.add_argument(
        "--tiny-mode",
        action="store_true",
        help="Ultra-light dataset profile for low-resource laptops.",
    )
    parser.add_argument(
        "--skip-deep",
        action="store_true",
        help="Skip deep-model training.",
    )
    parser.add_argument(
        "--skip-classical",
        action="store_true",
        help="Skip classical-model training.",
    )
    args = parser.parse_args()

    run_training(
        include_ptbxl=not args.no_ptbxl,
        force_rebuild_dataset=args.rebuild_dataset,
        deep_epochs=args.epochs,
        deep_batch_size=args.batch_size,
        quick_mode=args.quick_mode,
        tiny_mode=args.tiny_mode,
        skip_deep=args.skip_deep,
        skip_classical=args.skip_classical,
    )


if __name__ == "__main__":
    main()
