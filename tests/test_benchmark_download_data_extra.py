from pathlib import Path

from image_segmentation.benchmark.download_data import download_lvis_val, main


def test_download_data_lvis_dry_run(tmp_path, capsys):
    assert main(["--dataset", "lvis_val", "--output-dir", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "LVIS" in out
    assert "lvis_v1_val.json" in out
    assert not any(tmp_path.iterdir())


def test_download_lvis_val_dry_run(tmp_path, capsys):
    summary = download_lvis_val(str(tmp_path), dry_run=True)
    assert summary["annotations_file"].endswith("lvis/annotations/lvis_v1_val.json")
    assert "remote_file_size_bytes" in capsys.readouterr().out


def test_download_lvis_val_target_paths(tmp_path):
    summary = download_lvis_val(str(tmp_path), dry_run=True)
    assert summary["annotations_file"] == str(tmp_path / "lvis" / "annotations" / "lvis_v1_val.json")
    assert summary["images_dir"] == str(tmp_path / "coco" / "val2017")


def test_download_lvis_val_does_not_download_coco_images(tmp_path):
    download_lvis_val(str(tmp_path), dry_run=True)
    assert not (tmp_path / "coco").exists()


def test_download_lvis_val_existing_file_skip(tmp_path, capsys):
    target = tmp_path / "lvis" / "annotations" / "lvis_v1_val.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    summary = download_lvis_val(str(tmp_path))
    assert summary["annotations_file"] == str(target)
    assert "[SKIP]" in capsys.readouterr().out


def test_download_data_manual_dataset_explains_required_layout(tmp_path, capsys):
    assert main(["--dataset", "dis5k", "--output-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "requires manual download" in out
    assert "images" in out
    assert "masks" in out


def test_dis5k_download_dry_run_manual_instructions(tmp_path, capsys):
    assert main(["--dataset", "dis5k", "--output-dir", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "foreground/boundary stress" in out
    assert "configs/benchmark_v0_1.dis5k300.yaml" in out
    assert "images_dir" in out
    assert "masks_dir" in out
    assert list(Path(tmp_path).iterdir()) == []


def test_download_data_open_images_requires_fiftyone_when_enabled(tmp_path, capsys):
    assert (
        main(
            [
                "--dataset",
                "open_images_v7_segmentations",
                "--output-dir",
                str(tmp_path),
                "--use-fiftyone",
                "true",
            ]
        )
        == 1
    )
    assert "FiftyOne" in capsys.readouterr().out or "export wiring" in capsys.readouterr().out


def test_download_data_does_not_succeed_silently_for_manual_dataset(tmp_path):
    assert main(["--dataset", "dis5k", "--output-dir", str(tmp_path)]) != 0


def test_download_data_dry_run_no_files_written(tmp_path):
    assert main(["--dataset", "coco_val2017", "--output-dir", str(tmp_path), "--dry-run"]) == 0
    assert list(Path(tmp_path).iterdir()) == []


def test_cod10k_manual_download_message(tmp_path, capsys):
    assert main(["--dataset", "cod10k", "--output-dir", str(tmp_path)]) == 1
    assert "COD10K" in capsys.readouterr().out


def test_camo_manual_download_message(tmp_path, capsys):
    assert main(["--dataset", "camo", "--output-dir", str(tmp_path)]) == 1
    assert "CAMO" in capsys.readouterr().out


def test_cod10k_dry_run_accepts_target_root_and_writes_nothing(tmp_path, capsys):
    assert main(["--dataset", "cod10k", "--target-root", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "target_root" in out
    assert "benchmark_v0_1.cod10k.yaml" in out
    assert f"images_dir={tmp_path / 'images'}" in out
    assert f"masks_dir={tmp_path / 'masks'}" in out
    assert f"{tmp_path / 'cod10k' / 'images'}" not in out
    assert "[SOURCE]" in out
    assert "[ARCHIVE]" in out
    assert list(Path(tmp_path).iterdir()) == []


def test_camo_dry_run_accepts_target_root_and_reports_candidates(tmp_path, capsys):
    assert main(["--dataset", "camo", "--target-root", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"image_directory_candidates={['%s' % (tmp_path / 'images')]}" in out
    assert "quick_image_count=unavailable" in out
    assert "CC-BY-NC-SA" in out
