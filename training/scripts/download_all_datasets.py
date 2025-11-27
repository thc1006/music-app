#!/usr/bin/env python3
"""
主下載腳本 - 下載所有 OMR barline 數據集
"""
import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse


class DatasetDownloader:
    """數據集下載器"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, dest_path: Path, desc: str = "Downloading"):
        """下載文件並顯示進度條"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(dest_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            print(f"✓ 下載完成: {dest_path}")
            return True

        except Exception as e:
            print(f"✗ 下載失敗: {e}")
            return False

    def extract_archive(self, archive_path: Path, extract_to: Path):
        """解壓縮檔案"""
        print(f"解壓縮: {archive_path.name}")

        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz', '.bz2']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"✗ 不支援的壓縮格式: {archive_path.suffix}")
                return False

            print(f"✓ 解壓完成: {extract_to}")
            return True

        except Exception as e:
            print(f"✗ 解壓失敗: {e}")
            return False

    def download_omr_layout(self):
        """下載 OMR Layout Analysis 數據集"""
        print("\n" + "="*60)
        print("下載 OMR Layout Analysis Dataset")
        print("="*60)

        output_dir = self.base_dir / "omr_layout"
        output_dir.mkdir(exist_ok=True)

        # GitHub Release URL
        dataset_url = "https://github.com/v-dvorak/omr-layout-analysis/releases/download/Latest/datasets-release.zip"
        zip_path = output_dir / "datasets-release.zip"

        if zip_path.exists():
            print(f"檔案已存在: {zip_path}")
        else:
            success = self.download_file(dataset_url, zip_path, "OMR Layout Dataset")
            if not success:
                print("⚠ 自動下載失敗，請手動下載:")
                print(f"  URL: {dataset_url}")
                print(f"  目標: {zip_path}")
                return False

        # 解壓縮
        if not (output_dir / "datasets-release").exists():
            self.extract_archive(zip_path, output_dir)
        else:
            print("數據集已解壓")

        return True

    def download_audiolabs(self):
        """下載 AudioLabs v2 數據集"""
        print("\n" + "="*60)
        print("下載 AudioLabs v2 Dataset")
        print("="*60)

        output_dir = self.base_dir / "audiolabs"
        output_dir.mkdir(exist_ok=True)

        # AudioLabs 數據集 URL
        dataset_url = "https://www.audiolabs-erlangen.de/resources/MIR/2019-ISMIR-LBD-Measures/MeasureBoundingBoxAnnotations.zip"
        zip_path = output_dir / "audiolabs_measures.zip"

        if zip_path.exists():
            print(f"檔案已存在: {zip_path}")
        else:
            success = self.download_file(dataset_url, zip_path, "AudioLabs Dataset")
            if not success:
                print("⚠ 自動下載失敗，請手動下載:")
                print(f"  URL: {dataset_url}")
                print(f"  目標: {zip_path}")
                return False

        # 解壓縮
        extracted_dirs = ["MeasureBoundingBoxAnnotations", "2019_MeasureDetection_ISMIR2019"]
        if not any((output_dir / d).exists() for d in extracted_dirs):
            self.extract_archive(zip_path, output_dir)
        else:
            print("數據集已解壓")

        return True

    def download_doremi(self):
        """下載 DoReMi 數據集"""
        print("\n" + "="*60)
        print("下載 DoReMi Dataset")
        print("="*60)

        output_dir = self.base_dir / "doremi"
        output_dir.mkdir(exist_ok=True)

        # DoReMi GitHub Release URL
        dataset_url = "https://github.com/steinbergmedia/DoReMi/releases/download/v1.0/DoReMi_1.0.zip"
        zip_path = output_dir / "doremi_v1.0.zip"

        if zip_path.exists():
            print(f"檔案已存在: {zip_path}")
        else:
            success = self.download_file(dataset_url, zip_path, "DoReMi Dataset")
            if not success:
                print("⚠ 自動下載失敗，請手動下載:")
                print(f"  URL: {dataset_url}")
                print(f"  或訪問: https://github.com/steinbergmedia/DoReMi/releases")
                print(f"  目標: {zip_path}")
                return False

        # 解壓縮
        if not (output_dir / "DoReMi_1.0").exists():
            self.extract_archive(zip_path, output_dir)
        else:
            print("數據集已解壓")

        return True

    def download_all(self, datasets: list = None):
        """下載所有數據集"""
        if datasets is None:
            datasets = ['omr_layout', 'audiolabs', 'doremi']

        results = {}

        if 'omr_layout' in datasets:
            results['omr_layout'] = self.download_omr_layout()

        if 'audiolabs' in datasets:
            results['audiolabs'] = self.download_audiolabs()

        if 'doremi' in datasets:
            results['doremi'] = self.download_doremi()

        # 總結
        print("\n" + "="*60)
        print("下載總結")
        print("="*60)
        for dataset, success in results.items():
            status = "✓ 成功" if success else "✗ 失敗"
            print(f"{dataset}: {status}")

        return all(results.values())


def main():
    parser = argparse.ArgumentParser(description='下載 OMR barline 數據集')
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['omr_layout', 'audiolabs', 'doremi', 'all'],
        default=['all'],
        help='要下載的數據集 (預設: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/thc1006/dev/music-app/training/datasets/external_barlines',
        help='輸出目錄'
    )

    args = parser.parse_args()

    # 處理 'all' 選項
    if 'all' in args.datasets:
        datasets = ['omr_layout', 'audiolabs', 'doremi']
    else:
        datasets = args.datasets

    print("OMR Barline 數據集下載器")
    print(f"輸出目錄: {args.output}")
    print(f"數據集: {', '.join(datasets)}")

    downloader = DatasetDownloader(args.output)
    success = downloader.download_all(datasets)

    if success:
        print("\n✓ 所有數據集下載完成！")
        print("\n下一步:")
        print("  1. 運行 convert_omr_layout.py")
        print("  2. 運行 convert_audiolabs.py")
        print("  3. 運行 convert_doremi.py")
        print("  4. 運行 merge_barline_datasets.py")
        return 0
    else:
        print("\n⚠ 部分數據集下載失敗，請檢查上方訊息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
