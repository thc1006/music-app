#!/usr/bin/env python3
"""
即時訓練監控腳本 - 追蹤 GPU/CPU 使用率與訓練進度
"""
import time
import subprocess
import pandas as pd
from pathlib import Path
import sys

def get_gpu_stats():
    """獲取 GPU 使用狀態"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(stats[0]),
                'mem_util': int(stats[1]),
                'mem_used': int(stats[2]),
                'mem_total': int(stats[3]),
                'temp': int(stats[4]),
                'power': float(stats[5])
            }
    except Exception as e:
        return None

def get_cpu_stats():
    """獲取 CPU 使用率"""
    try:
        result = subprocess.run(['top', '-bn1'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Cpu(s)' in line:
                # 提取使用率
                parts = line.split(',')
                idle = float(parts[3].split()[0])
                return 100 - idle
    except:
        return None

def read_training_progress(results_csv):
    """讀取訓練進度"""
    try:
        df = pd.read_csv(results_csv)
        if len(df) > 0:
            latest = df.iloc[-1]
            return {
                'epoch': int(latest['epoch']) + 1,
                'train_loss': float(latest['train/box_loss']) if 'train/box_loss' in latest else 0,
                'val_map50': float(latest['metrics/mAP50(B)']) if 'metrics/mAP50(B)' in latest else 0,
                'val_map50_95': float(latest['metrics/mAP50-95(B)']) if 'metrics/mAP50-95(B)' in latest else 0,
            }
    except:
        return None

def monitor_training(project_dir, name, interval=5):
    """即時監控訓練"""
    results_csv = Path(project_dir) / name / 'results.csv'

    print("\n" + "="*80)
    print("  🖥️  Phase 10.1 即時訓練監控")
    print("="*80)
    print(f"\n監控文件: {results_csv}")
    print(f"更新間隔: {interval} 秒")
    print(f"\n按 Ctrl+C 停止監控\n")

    last_epoch = -1
    best_map = 0

    try:
        while True:
            # GPU 狀態
            gpu = get_gpu_stats()
            cpu = get_cpu_stats()

            # 訓練進度
            progress = read_training_progress(results_csv)

            # 清空螢幕（可選）
            # print("\033[2J\033[H", end='')

            print(f"\r[{time.strftime('%H:%M:%S')}] ", end='')

            if gpu:
                print(f"GPU: {gpu['gpu_util']:3d}% | VRAM: {gpu['mem_used']:5d}/{gpu['mem_total']:5d} MB ({gpu['mem_util']:3d}%) | "
                      f"Temp: {gpu['temp']:2d}°C | Power: {gpu['power']:5.1f}W", end=' | ')

            if cpu:
                print(f"CPU: {cpu:5.1f}%", end=' | ')

            if progress:
                epoch = progress['epoch']
                if epoch > last_epoch:
                    last_epoch = epoch
                    if progress['val_map50_95'] > best_map:
                        best_map = progress['val_map50_95']

                print(f"Epoch: {epoch:3d}/150 | Loss: {progress['train_loss']:.4f} | "
                      f"mAP50-95: {progress['val_map50_95']:.4f} | mAP50: {progress['val_map50']:.4f} | "
                      f"Best: {best_map:.4f}", end='')
            else:
                print("等待訓練開始...", end='')

            sys.stdout.flush()
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n監控已停止")

if __name__ == "__main__":
    project_dir = "/home/thc1006/dev/music-app/training/harmony_omr_v2_phase10_1"
    name = "phase10_1_training"

    monitor_training(project_dir, name, interval=5)
