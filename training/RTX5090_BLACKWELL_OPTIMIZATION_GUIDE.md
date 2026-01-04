# RTX 5090 Blackwell 深度學習優化指南

## 創建日期: 2025-12-20
## 適用工作站: i9-14900 + RTX 5090 + 125GB RAM

---

## 1. 系統規格分析

### 1.1 硬體配置

| 組件 | 規格 | 優化要點 |
|------|------|---------|
| **GPU** | RTX 5090 (33.7GB VRAM) | Blackwell 架構 sm_120 |
| **CPU** | i9-14900 (24 核心) | 8 P-cores + 16 E-cores |
| **RAM** | 125 GB DDR5 | 足夠緩存整個數據集 |
| **CUDA** | 12.8 | Blackwell 最低需求 |
| **cuDNN** | 9.1 | 最新版本 |
| **PyTorch** | 2.9.1 | 原生 sm_120 支援 |

### 1.2 Blackwell 架構特性

RTX 5090 採用 NVIDIA Blackwell 架構 (sm_120)，具有以下特性：

1. **第五代 Tensor Cores**
   - 原生 FP8 (8-bit) 支援
   - 增強 BF16 和 FP16 性能
   - TF32 自動加速

2. **21,760 CUDA Cores**
   - 比 RTX 4090 增加 33%

3. **32GB GDDR7 VRAM @ 28 Gbps**
   - 頻寬提升 50%+

4. **功耗管理**
   - TDP 575W
   - 建議 1000W PSU

---

## 2. 優化策略詳解

### 2.1 Blackwell 專屬優化

#### 2.1.1 TF32 (Tensor Float 32)

```python
# 啟用 TF32 - Blackwell 原生支援
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**效果**: 矩陣運算速度提升 3-5x，精度損失可忽略

#### 2.1.2 BF16 (Brain Float 16)

```python
# 使用 BF16 代替 FP16 - 更穩定
if torch.cuda.is_bf16_supported():
    model = model.to(dtype=torch.bfloat16)
```

**BF16 vs FP16**:
| 特性 | FP16 | BF16 |
|------|------|------|
| 指數位數 | 5 | 8 |
| 尾數位數 | 10 | 7 |
| 動態範圍 | ±65,504 | 同 FP32 |
| 穩定性 | 需 loss scaling | 不需要 |
| 精度 | 較高 | 略低 |

**結論**: BF16 更適合訓練，不易出現 inf/nan

#### 2.1.3 torch.compile

```python
# PyTorch 2.x JIT 編譯優化
model = torch.compile(model, mode="reduce-overhead")
```

**可用模式**:
- `default`: 平衡編譯時間和性能
- `reduce-overhead`: 最小化 Python 開銷
- `max-autotune`: 最大性能（編譯時間長）

#### 2.1.4 CUDA 記憶體分配器優化

```python
# PyTorch 2.9+ 使用新的環境變數名稱
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
```

**參數說明**:
- `max_split_size_mb`: 減少記憶體碎片
- `expandable_segments`: 動態擴展記憶體段

### 2.2 多核心 CPU 優化

#### 2.2.1 DataLoader workers

```python
# i9-14900 (24 核心) 最佳配置
workers = 20  # 預留 4 核給系統
prefetch_factor = 4  # 每 worker 預取 4 批
pin_memory = True  # 鎖定記憶體
persistent_workers = True  # 保持進程
```

**workers 選擇指南**:
| CPU 核心數 | 建議 workers |
|-----------|-------------|
| 4-8 | 4 |
| 8-16 | 8-12 |
| 16-24 | 16-20 |
| 24+ | 20-24 |

#### 2.2.2 pin_memory

```python
pin_memory = True  # 對 GPU 訓練必須啟用
```

**原理**: 使用 Page-locked (pinned) 記憶體，允許 GPU 直接存取 CPU 記憶體，跳過中間複製。

**效果**: CPU→GPU 傳輸速度提升 2-3x

#### 2.2.3 persistent_workers

```python
persistent_workers = True
```

**原理**: 保持 worker 進程存活，避免每個 epoch 重新創建進程。

**效果**: 每個 epoch 開始時節省 5-10 秒

### 2.3 大容量 RAM 優化

#### 2.3.1 數據集緩存

```python
# 125GB RAM 足夠緩存整個數據集
cache = 'ram'  # 或 'disk' 或 False
```

**緩存選項**:
| 選項 | 速度 | RAM 需求 | 適用場景 |
|------|------|---------|---------|
| `ram` | 最快 | 高 | RAM > 數據集 2x |
| `disk` | 中等 | 低 | SSD + 有限 RAM |
| `False` | 最慢 | 無 | 超大數據集 |

**我們的情況**: 125GB RAM，數據集約 10-20GB → 使用 `ram`

### 2.4 批次大小優化

#### 2.4.1 自動偵測

```python
# YOLO ultralytics 支援自動偵測
batch = -1  # 自動計算最佳批次大小
```

#### 2.4.2 手動計算

根據 VRAM 容量:
| VRAM | YOLO12s @ 640px | YOLO12m @ 640px |
|------|-----------------|-----------------|
| 8GB | 8 | 4 |
| 12GB | 16 | 8 |
| 24GB | 24-32 | 16-24 |
| **32GB** | **32-48** | **24-32** |

**RTX 5090 (32GB) 建議**: batch=32 (YOLO12s @ 640px)

---

## 3. YOLO 訓練專屬優化

### 3.1 混合精度訓練 (AMP)

```python
amp = True  # 自動混合精度
```

**效果**:
- 記憶體使用減少 30-50%
- 訓練速度提升 20-40%
- 精度損失 < 0.5%

### 3.2 OMR 專用數據增強

```python
# 樂譜特性優化
fliplr = 0.0  # 不左右翻轉（音符有方向性）
flipud = 0.0  # 不上下翻轉
degrees = 0.0  # 不旋轉（五線譜是水平的）
perspective = 0.0  # 不透視變換
mosaic = 0.5  # 適度使用
mixup = 0.1  # 少量使用
```

### 3.3 學習率調度

```python
lr0 = 0.0005  # 蒸餾用較低初始學習率
lrf = 0.01    # 最終學習率 = lr0 * lrf
warmup_epochs = 3  # 預熱
```

---

## 4. 環境變數配置

```bash
# .bashrc 或執行前設置 (PyTorch 2.9+)
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDA_ARCH_LIST=12.0
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
```

---

## 5. 性能預期

### 5.1 訓練速度對比

| 配置 | 批次大小 | Workers | 緩存 | 預計速度 |
|------|---------|---------|------|---------|
| 未優化 | 16 | 8 | disk | 1x |
| 基礎優化 | 24 | 16 | ram | 1.5x |
| **完全優化** | **32** | **20** | **ram** | **2-2.5x** |

### 5.2 記憶體使用

| 項目 | 未優化 | 優化後 |
|------|--------|--------|
| GPU VRAM | ~20GB | ~28GB (更好利用) |
| CPU RAM | ~30GB | ~50GB (緩存數據集) |
| I/O 等待 | 高 | 極低 |

---

## 6. 監控與調試

### 6.1 GPU 監控

```bash
# 即時監控
watch -n 1 nvidia-smi

# 詳細信息
nvidia-smi dmon -s pucvmet -d 1
```

### 6.2 訓練監控

```python
# 使用 tensorboard
tensorboard --logdir ./runs
```

### 6.3 記憶體分析

```python
# PyTorch 記憶體分析
torch.cuda.memory_summary(device=0, abbreviated=False)
```

---

## 7. 故障排除

### 7.1 CUDA Out of Memory

```python
# 1. 減少批次大小
batch = 24  # 從 32 降到 24

# 2. 清理緩存
torch.cuda.empty_cache()
gc.collect()

# 3. 使用梯度累積
accumulate = 2  # 等效於 batch*2
```

### 7.2 DataLoader 速度慢

1. 檢查 workers 數量
2. 確認 pin_memory=True
3. 使用 SSD 而非 HDD
4. 啟用數據集緩存

### 7.3 訓練不穩定

1. 降低學習率: lr0 = 0.0001
2. 增加 warmup: warmup_epochs = 5
3. 使用 BF16 代替 FP16
4. 檢查數據集標註品質

---

## 8. 參考資源

- [NVIDIA Blackwell RTX GPU Migration Guide](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus)
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Ultralytics YOLO Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)
- [PyTorch DataLoader Optimization](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8)
- [RTX 5090 PyTorch Forums](https://discuss.pytorch.org/t/nvidia-geforce-rtx-5090/218954)

---

**文檔版本**: 1.0
**更新日期**: 2025-12-20
