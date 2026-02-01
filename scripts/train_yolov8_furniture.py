#!/usr/bin/env python3
"""
YOLOv8 Furniture Detection Training Script

This script merges suitable Hugging Face furniture datasets and trains YOLOv8.
Supports:
- YOLO-ready datasets (with data.yaml)
- COCO/Parquet datasets (auto-converted to YOLO format)

Usage:
    python scripts/train_yolov8_furniture.py

Environment variables:
    HF_TOKEN: Hugging Face token for private/gated datasets
    WORKDIR: Working directory (default: ./furniture_yolo_training)
    EPOCHS: Number of training epochs (default: 60)
    BATCH_SIZE: Training batch size (default: 16)
    IMG_SIZE: Image size for training (default: 640)
    MODEL: YOLOv8 model to use (default: yolov8n.pt)
    DEVICE: Training device (default: 0 for GPU)
"""

import os
import sys
import glob
import yaml
import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics not installed. Run: pip install ultralytics==8.2.0")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None
    print("Warning: huggingface_hub not installed. Run: pip install huggingface-hub")

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
    print("Warning: datasets not installed. Run: pip install datasets")

try:
    from PIL import Image
except ImportError:
    Image = None
    print("Warning: pillow not installed. Run: pip install pillow")


# -------------------- CONFIGURATION --------------------

class YOLOv8TrainingConfig:
    """Configuration for YOLOv8 furniture detection training."""

    def __init__(self):
        # Hugging Face token (for gated/private datasets)
        self.hf_token = os.getenv("HF_TOKEN", "")

        # Directory settings
        self.workdir = os.getenv("WORKDIR", "./furniture_yolo_training")
        self.raw_dir = os.path.join(self.workdir, "raw")
        self.build_dir = os.path.join(self.workdir, "build")
        self.merged_dir = os.path.join(self.workdir, "merged_yolo")
        self.runs_dir = os.path.join(self.workdir, "runs")

        # Final unified class names
        self.final_names = ["chair", "sofa", "table"]

        # Training parameters
        self.model = os.getenv("MODEL", "yolov8n.pt")
        self.img_size = int(os.getenv("IMG_SIZE", 640))
        self.epochs = int(os.getenv("EPOCHS", 60))
        self.batch_size = int(os.getenv("BATCH_SIZE", 16))
        self.device = os.getenv("DEVICE", "0")  # GPU device
        self.patience = int(os.getenv("PATIENCE", 20))
        self.cache = os.getenv("CACHE", "true").lower() == "true"

        # Default datasets
        self.yolo_ready_datasets = [
            "Libre-YOLO/furniture-ngpea",
        ]
        self.coco_parquet_datasets = [
            "Francesco/furniture-ngpea",
        ]

    def ensure_directories(self):
        """Create all required directories."""
        for path in [self.raw_dir, self.build_dir, self.merged_dir, self.runs_dir]:
            os.makedirs(path, exist_ok=True)
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.merged_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.merged_dir, "labels", split), exist_ok=True)


# -------------------- UTILITY FUNCTIONS --------------------

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def write_yaml(path: str, data: Dict) -> None:
    """Write data to a YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def read_yaml(path: str) -> Dict:
    """Read data from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(folder: str) -> List[str]:
    """List all image files in a folder."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for ext in exts:
        images += glob.glob(os.path.join(folder, ext))
    return sorted(images)


def find_data_yaml(root: str) -> Optional[str]:
    """Find data.yaml file in a directory tree."""
    candidates = glob.glob(os.path.join(root, "**", "data.yaml"), recursive=True)
    return candidates[0] if candidates else None


def normalize_names(names: Any) -> Optional[List[str]]:
    """Normalize class names to lowercase list."""
    if names is None:
        return None
    if isinstance(names, list):
        return [str(x).strip().lower() for x in names]
    if isinstance(names, dict):
        items = sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])
        return [str(v).strip().lower() for _, v in items]
    return None


def infer_label_dir(img_dir: str) -> str:
    """Infer labels directory from images directory."""
    # YOLO typical: images/train -> labels/train
    if "/images/" in img_dir:
        return img_dir.replace("/images/", "/labels/")
    return os.path.join(os.path.dirname(img_dir), "labels")


def remap_label_file(in_path: str, out_path: str, id_map: Dict[int, int]) -> None:
    """
    Remap class IDs in a YOLO label file.

    Args:
        in_path: Input label file path
        out_path: Output label file path
        id_map: Mapping from old class ID to new class ID
    """
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    out_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        if cls_id not in id_map:
            continue
        parts[0] = str(id_map[cls_id])
        out_lines.append(" ".join(parts))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))


# -------------------- DATASET IMPORT FUNCTIONS --------------------

class DatasetImporter:
    """Handles importing datasets from various sources."""

    def __init__(self, config: YOLOv8TrainingConfig):
        self.config = config

    def import_yolo_ready(self, repo_id: str) -> Dict:
        """
        Import a YOLO-ready dataset from Hugging Face.

        Args:
            repo_id: Hugging Face repository ID (e.g., "Libre-YOLO/furniture-ngpea")

        Returns:
            Dict with dataset info and paths
        """
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is required for importing YOLO datasets")

        local_dir = os.path.join(
            self.config.raw_dir, repo_id.replace("/", "__")
        )

        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)

        print(f"📥 Downloading YOLO-ready dataset: {repo_id}")
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=self.config.hf_token if self.config.hf_token else None,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        # Find and parse data.yaml
        data_yaml_path = find_data_yaml(local_dir)
        if not data_yaml_path:
            raise RuntimeError(f"{repo_id}: data.yaml not found (not YOLO-ready)")

        data_yaml = read_yaml(data_yaml_path)
        names = normalize_names(data_yaml.get("names"))
        if not names:
            raise RuntimeError(f"{repo_id}: names not found in data.yaml")

        # Resolve base path
        base = data_yaml.get("path", os.path.dirname(data_yaml_path))
        if not os.path.isabs(base):
            base = os.path.join(os.path.dirname(data_yaml_path), base)

        def abs_dir(x: str) -> str:
            return x if os.path.isabs(x) else os.path.join(base, x)

        train_img = abs_dir(data_yaml["train"])
        val_img = abs_dir(data_yaml["val"])
        test_img = abs_dir(data_yaml.get("test", "")) if data_yaml.get("test") else None

        train_lbl = infer_label_dir(train_img)
        val_lbl = infer_label_dir(val_img)
        test_lbl = infer_label_dir(test_img) if test_img else None

        print(f"✅ Imported {repo_id}: {len(names)} classes")

        return {
            "repo": repo_id,
            "names": names,
            "train_img": train_img,
            "val_img": val_img,
            "test_img": test_img,
            "train_lbl": train_lbl,
            "val_lbl": val_lbl,
            "test_lbl": test_lbl,
        }

    def convert_coco_parquet_to_yolo(self, repo_id: str) -> Dict:
        """
        Convert a COCO/Parquet dataset from Hugging Face to YOLO format.

        Args:
            repo_id: Hugging Face repository ID (e.g., "Francesco/furniture-ngpea")

        Returns:
            Dict with dataset info and paths
        """
        if load_dataset is None:
            raise RuntimeError("datasets library is required for COCO conversion")
        if Image is None:
            raise RuntimeError("pillow is required for COCO conversion")

        out_dir = os.path.join(
            self.config.build_dir, repo_id.replace("/", "__"), "yolo"
        )

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        for split in ["train", "validation", "test"]:
            ensure_dir(os.path.join(out_dir, "images", split))
            ensure_dir(os.path.join(out_dir, "labels", split))

        print(f"📥 Loading COCO/Parquet dataset: {repo_id}")
        ds = load_dataset(repo_id)

        # Try to get class names from dataset features
        class_names = None
        try:
            feats = ds[list(ds.keys())[0]].features
            if "objects" in feats and hasattr(feats["objects"], "feature"):
                obj_feat = feats["objects"].feature
                if "category" in obj_feat and hasattr(obj_feat["category"], "names"):
                    class_names = [n.strip().lower() for n in obj_feat["category"].names]
        except Exception:
            pass

        # Fallback to final names
        if not class_names:
            class_names = self.config.final_names[:]

        # Build class mapping: local -> final
        local_to_final = {}
        for i, name in enumerate(class_names):
            name_lower = name.strip().lower()
            if name_lower in self.config.final_names:
                local_to_final[i] = self.config.final_names.index(name_lower)

        def get_objects(example: Dict) -> Tuple[Optional[str], Optional[Dict]]:
            """Extract objects from example."""
            for key in ["objects", "annotations"]:
                if key in example:
                    return key, example[key]
            return None, None

        def yolo_line(cls: int, x: float, y: float, w: float, h: float,
                      img_w: int, img_h: int) -> str:
            """Create YOLO format line from COCO bbox."""
            # COCO bbox: [x, y, width, height] in pixels
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width_norm = w / img_w
            height_norm = h / img_h
            return f"{cls} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"

        # Process each split
        for split_name in ds.keys():
            split = split_name
            print(f"  Converting {repo_id} {split_name}...")

            for idx in tqdm(range(len(ds[split_name])), desc=f"    {split_name}"):
                example = ds[split_name][idx]

                # Get image
                img = example.get("image")
                if img is None:
                    continue

                if isinstance(img, Image.Image):
                    pil_img = img
                else:
                    pil_img = Image.fromarray(img)

                img_w, img_h = pil_img.size
                stem = f"{idx:08d}"

                img_out = os.path.join(out_dir, "images", split, f"{stem}.jpg")
                lbl_out = os.path.join(out_dir, "labels", split, f"{stem}.txt")

                pil_img.convert("RGB").save(img_out, quality=95)

                _, objs = get_objects(example)
                if objs is None:
                    open(lbl_out, "w").write("")
                    continue

                # Get bboxes and categories
                bboxes = objs.get("bbox") if isinstance(objs, dict) else None
                cats = objs.get("category") if isinstance(objs, dict) else None

                if bboxes is None or cats is None:
                    open(lbl_out, "w").write("")
                    continue

                lines = []
                for bbox, cat in zip(bboxes, cats):
                    x, y, w, h = bbox
                    if int(cat) not in local_to_final:
                        continue
                    cls_id = local_to_final[int(cat)]
                    # Skip tiny boxes
                    if w * h < 20:
                        continue
                    lines.append(yolo_line(cls_id, x, y, w, h, img_w, img_h))

                with open(lbl_out, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))

        # Create YOLO data.yaml
        data_yaml = {
            "path": out_dir,
            "train": "images/train",
            "val": "images/validation" if os.path.isdir(
                os.path.join(out_dir, "images", "validation")
            ) else "images/val",
            "test": "images/test" if os.path.isdir(
                os.path.join(out_dir, "images", "test")
            ) else None,
            "names": {i: n for i, n in enumerate(self.config.final_names)},
        }
        write_yaml(os.path.join(out_dir, "data.yaml"), data_yaml)

        print(f"✅ Converted {repo_id} to YOLO format")

        val_dir = "validation" if os.path.isdir(
            os.path.join(out_dir, "images", "validation")
        ) else "val"

        return {
            "repo": repo_id,
            "names": self.config.final_names[:],
            "train_img": os.path.join(out_dir, "images", "train"),
            "val_img": os.path.join(out_dir, "images", val_dir),
            "test_img": os.path.join(out_dir, "images", "test") if os.path.isdir(
                os.path.join(out_dir, "images", "test")
            ) else None,
            "train_lbl": os.path.join(out_dir, "labels", "train"),
            "val_lbl": os.path.join(out_dir, "labels", val_dir),
            "test_lbl": os.path.join(out_dir, "labels", "test") if os.path.isdir(
                os.path.join(out_dir, "labels", "test")
            ) else None,
        }


# -------------------- DATASET MERGER --------------------

class DatasetMerger:
    """Merges multiple datasets into a single YOLO dataset."""

    def __init__(self, config: YOLOv8TrainingConfig):
        self.config = config
        self.train_count = 0
        self.val_count = 0
        self.test_count = 0

    def merge_split(
        self,
        info: Dict,
        split: str,
        in_img_dir: Optional[str],
        in_lbl_dir: Optional[str],
        out_img_dir: str,
        out_lbl_dir: str,
    ) -> Tuple[int, int]:
        """
        Merge a single split from a dataset.

        Returns:
            Tuple of (kept_count, missing_count)
        """
        if not in_img_dir or not os.path.isdir(in_img_dir):
            return 0, 0

        images = list_images(in_img_dir)
        if not images:
            return 0, 0

        # Build class ID mapping
        local_names = [n.strip().lower() for n in info["names"]]
        id_map = {}
        for i, name in enumerate(local_names):
            if name in self.config.final_names:
                id_map[i] = self.config.final_names.index(name)

        kept = 0
        missing = 0

        for img_path in tqdm(images, desc=f"  Merging {info['repo']} {split}", leave=False):
            stem = Path(img_path).stem
            lbl_path = os.path.join(in_lbl_dir, stem + ".txt")

            if not os.path.exists(lbl_path):
                missing += 1
                continue

            # Generate unique filename
            if split == "train":
                new_name = f"{info['repo'].replace('/', '_')}_{self.train_count:08d}"
                self.train_count += 1
            elif split == "val":
                new_name = f"{info['repo'].replace('/', '_')}_{self.val_count:08d}"
                self.val_count += 1
            else:
                new_name = f"{info['repo'].replace('/', '_')}_{self.test_count:08d}"
                self.test_count += 1

            ext = os.path.splitext(img_path)[1].lower()
            out_img = os.path.join(out_img_dir, new_name + ext)
            out_lbl = os.path.join(out_lbl_dir, new_name + ".txt")

            shutil.copy2(img_path, out_img)
            remap_label_file(lbl_path, out_lbl, id_map)

            # Remove empty labels
            if os.path.getsize(out_lbl) == 0:
                os.remove(out_lbl)
                os.remove(out_img)
                if split == "train":
                    self.train_count -= 1
                elif split == "val":
                    self.val_count -= 1
                else:
                    self.test_count -= 1
                continue

            kept += 1

        return kept, missing

    def merge_datasets(self, datasets: List[Dict]) -> str:
        """
        Merge all datasets into the merged directory.

        Returns:
            Path to the merged data.yaml
        """
        for info in datasets:
            # Merge train
            kt, mt = self.merge_split(
                info, "train",
                info.get("train_img"), info.get("train_lbl"),
                os.path.join(self.config.merged_dir, "images", "train"),
                os.path.join(self.config.merged_dir, "labels", "train"),
            )

            # Merge val
            kv, mv = self.merge_split(
                info, "val",
                info.get("val_img"), info.get("val_lbl"),
                os.path.join(self.config.merged_dir, "images", "val"),
                os.path.join(self.config.merged_dir, "labels", "val"),
            )

            # Merge test (if available)
            ks, ms = 0, 0
            if info.get("test_img") and info.get("test_lbl"):
                ks, ms = self.merge_split(
                    info, "test",
                    info.get("test_img"), info.get("test_lbl"),
                    os.path.join(self.config.merged_dir, "images", "test"),
                    os.path.join(self.config.merged_dir, "labels", "test"),
                )

            print(
                f"✅ {info['repo']} merged -> "
                f"train:{kt} val:{kv} test:{ks} | "
                f"missing labels -> train:{mt} val:{mv} test:{ms}"
            )

        # Write merged data.yaml
        data_yaml = {
            "path": self.config.merged_dir,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: n for i, n in enumerate(self.config.final_names)},
        }
        data_yaml_path = os.path.join(self.config.merged_dir, "data.yaml")
        write_yaml(data_yaml_path, data_yaml)

        return data_yaml_path


# -------------------- TRAINING --------------------

class YOLOv8Trainer:
    """Handles YOLOv8 training."""

    def __init__(self, config: YOLOv8TrainingConfig):
        self.config = config

    def train(self, data_yaml_path: str) -> str:
        """
        Train YOLOv8 on the merged dataset.

        Returns:
            Path to the best weights
        """
        if YOLO is None:
            raise RuntimeError("ultralytics is required for training")

        # Check if dataset is valid
        train_images = glob.glob(
            os.path.join(self.config.merged_dir, "images", "train", "*")
        )
        if not train_images:
            raise RuntimeError("❌ Merged dataset is empty. Check dataset paths/labels.")

        print(f"\n{'='*60}")
        print("Starting YOLOv8 Training")
        print(f"{'='*60}")
        print(f"Model: {self.config.model}")
        print(f"Image size: {self.config.img_size}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Device: {self.config.device}")
        print(f"Train images: {len(train_images)}")
        print(f"{'='*60}\n")

        # Initialize and train model
        model = YOLO(self.config.model)
        model.train(
            data=data_yaml_path,
            imgsz=self.config.img_size,
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            device=self.config.device,
            project=self.config.runs_dir,
            name="furniture_detector",
            patience=self.config.patience,
            cache=self.config.cache,
        )

        best_weights = os.path.join(
            self.config.runs_dir, "furniture_detector", "weights", "best.pt"
        )
        return best_weights

    def validate(self, weights_path: str, sample_image: Optional[str] = None) -> None:
        """Run validation/prediction on a sample image."""
        if YOLO is None:
            return

        if sample_image is None:
            val_images = glob.glob(
                os.path.join(self.config.merged_dir, "images", "val", "*")
            )
            if val_images:
                sample_image = val_images[0]

        if sample_image:
            print(f"\n🔍 Running prediction on sample image: {sample_image}")
            model = YOLO(weights_path)
            model.predict(
                source=sample_image,
                imgsz=self.config.img_size,
                conf=0.25,
                save=True,
                project=os.path.join(self.config.runs_dir, "pred"),
                name="preview",
            )
            print(f"🖼️ Saved prediction to: {self.config.runs_dir}/pred/preview")


# -------------------- MAIN ENTRY POINT --------------------

def main():
    """Main entry point for YOLOv8 furniture training."""
    print("=" * 60)
    print("YOLOv8 Furniture Detection Training")
    print("=" * 60)

    # Initialize configuration
    config = YOLOv8TrainingConfig()
    config.ensure_directories()

    print("\n📁 Working directory:", config.workdir)
    print("📦 Model:", config.model)
    print("🎯 Classes:", config.final_names)
    print("⚙️ Epochs:", config.epochs)
    print("📊 Batch size:", config.batch_size)
    print("🖼️ Image size:", config.img_size)

    # Import datasets
    print("\n" + "=" * 60)
    print("Importing Datasets")
    print("=" * 60)

    importer = DatasetImporter(config)
    all_datasets = []

    # Import YOLO-ready datasets
    print("\n📥 YOLO-ready datasets:")
    for repo_id in config.yolo_ready_datasets:
        try:
            info = importer.import_yolo_ready(repo_id)
            all_datasets.append(info)
        except Exception as e:
            print(f"⚠️ Failed to import {repo_id}: {e}")

    # Convert COCO/Parquet datasets
    print("\n📥 COCO/Parquet datasets (converting to YOLO):")
    for repo_id in config.coco_parquet_datasets:
        try:
            info = importer.convert_coco_parquet_to_yolo(repo_id)
            all_datasets.append(info)
        except Exception as e:
            print(f"⚠️ Failed to convert {repo_id}: {e}")

    if not all_datasets:
        print("\n❌ No datasets were successfully imported!")
        return 1

    # Merge datasets
    print("\n" + "=" * 60)
    print("Merging Datasets")
    print("=" * 60)

    merger = DatasetMerger(config)
    data_yaml_path = merger.merge_datasets(all_datasets)

    # Print merge summary
    train_count = len(glob.glob(os.path.join(config.merged_dir, "images", "train", "*")))
    val_count = len(glob.glob(os.path.join(config.merged_dir, "images", "val", "*")))
    test_count = len(glob.glob(os.path.join(config.merged_dir, "images", "test", "*")))

    print(f"\n✅ Merge Complete!")
    print(f"   Train images: {train_count}")
    print(f"   Val images:   {val_count}")
    print(f"   Test images:  {test_count}")
    print(f"   data.yaml:    {data_yaml_path}")

    # Train model
    print("\n" + "=" * 60)
    print("Training YOLOv8")
    print("=" * 60)

    trainer = YOLOv8Trainer(config)
    best_weights = trainer.train(data_yaml_path)

    print(f"\n✅ Training Complete!")
    print(f"📦 Best weights: {best_weights}")

    # Run validation
    trainer.validate(best_weights)

    print("\n" + "=" * 60)
    print("✅ YOLOv8 Furniture Detection Training Completed!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
