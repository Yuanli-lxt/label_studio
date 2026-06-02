from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetRegistryEntry:
    dataset_name: str
    display_name: str
    task_type: str
    loader_type: str
    download_mode: str
    expected_local_dirs: tuple[str, ...]
    expected_annotation_files: tuple[str, ...]
    license_note: str
    official_page_note: str
    supports_scripted_download: bool
    supports_manual_import: bool
    supports_preflight: bool
    recommended_first_sample_size: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expected_local_dirs"] = list(self.expected_local_dirs)
        data["expected_annotation_files"] = list(self.expected_annotation_files)
        return data


DATASET_REGISTRY: dict[str, DatasetRegistryEntry] = {
    "coco_val2017": DatasetRegistryEntry(
        dataset_name="coco_val2017",
        display_name="COCO val2017",
        task_type="instance_segmentation",
        loader_type="coco",
        download_mode="scripted",
        expected_local_dirs=("data/external/coco/val2017",),
        expected_annotation_files=("data/external/coco/annotations/instances_val2017.json",),
        license_note="COCO data is subject to COCO image and annotation terms.",
        official_page_note="https://cocodataset.org/#download",
        supports_scripted_download=True,
        supports_manual_import=True,
        supports_preflight=True,
        recommended_first_sample_size=100,
    ),
    "lvis_val": DatasetRegistryEntry(
        dataset_name="lvis_val",
        display_name="LVIS v1 val",
        task_type="instance_segmentation",
        loader_type="lvis",
        download_mode="scripted_annotations_manual_or_coco_images",
        expected_local_dirs=("data/external/coco/val2017",),
        expected_annotation_files=("data/external/lvis/annotations/lvis_v1_val.json",),
        license_note="LVIS annotations are CC BY 4.0; images follow COCO/Flickr source terms.",
        official_page_note="https://www.lvisdataset.org/",
        supports_scripted_download=True,
        supports_manual_import=True,
        supports_preflight=True,
        recommended_first_sample_size=500,
    ),
    "dis5k": DatasetRegistryEntry(
        dataset_name="dis5k",
        display_name="DIS5K",
        task_type="binary_foreground_segmentation",
        loader_type="mask_folder",
        download_mode="manual_or_external_archive",
        expected_local_dirs=("data/external/dis5k/images", "data/external/dis5k/masks"),
        expected_annotation_files=(),
        license_note="Confirm DIS5K license and source terms before local use.",
        official_page_note="Manual placement is required for the first implementation.",
        supports_scripted_download=False,
        supports_manual_import=True,
        supports_preflight=True,
        recommended_first_sample_size=300,
    ),
    "cod10k": DatasetRegistryEntry(
        dataset_name="cod10k",
        display_name="COD10K",
        task_type="camouflaged_object_segmentation",
        loader_type="mask_folder",
        download_mode="manual_or_external_archive",
        expected_local_dirs=("data/external/cod10k/images", "data/external/cod10k/masks"),
        expected_annotation_files=(),
        license_note="Confirm COD10K license and source terms before local use; common mirrors list the license as unknown.",
        official_page_note="Official project/paper links commonly use Google Drive/Baidu-style hosting and are not scripted.",
        supports_scripted_download=False,
        supports_manual_import=True,
        supports_preflight=True,
        recommended_first_sample_size=300,
    ),
    "camo": DatasetRegistryEntry(
        dataset_name="camo",
        display_name="CAMO",
        task_type="camouflaged_object_segmentation",
        loader_type="mask_folder",
        download_mode="manual_or_external_archive",
        expected_local_dirs=("data/external/camo/images", "data/external/camo/masks"),
        expected_annotation_files=(),
        license_note="CAMO official page states CC-BY-NC-SA 4.0; confirm terms before local/commercial use.",
        official_page_note="Official CAMO project page provides manual download instructions; unstable Google Drive/Baidu links are not scripted.",
        supports_scripted_download=False,
        supports_manual_import=True,
        supports_preflight=True,
        recommended_first_sample_size=300,
    ),
    "open_images_v7_segmentations": DatasetRegistryEntry(
        dataset_name="open_images_v7_segmentations",
        display_name="Open Images V7 Segmentations",
        task_type="instance_segmentation",
        loader_type="open_images",
        download_mode="optional_fiftyone_subset",
        expected_local_dirs=("data/external/open_images_v7/validation/images", "data/external/open_images_v7/validation/segmentations"),
        expected_annotation_files=("data/external/open_images_v7/validation/annotations.jsonl",),
        license_note="Open Images assets have their own image-level licenses; download only subsets you are allowed to use.",
        official_page_note="https://storage.googleapis.com/openimages/web/index.html",
        supports_scripted_download=True,
        supports_manual_import=True,
        supports_preflight=True,
        recommended_first_sample_size=500,
    ),
}


def get_dataset_entry(dataset_name: str) -> DatasetRegistryEntry:
    try:
        return DATASET_REGISTRY[dataset_name]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"unknown dataset {dataset_name!r}; expected one of: {known}") from exc


def registry_as_dict() -> dict[str, dict[str, Any]]:
    return {name: entry.to_dict() for name, entry in DATASET_REGISTRY.items()}
