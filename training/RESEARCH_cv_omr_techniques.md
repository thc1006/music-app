# OMR Classical CV 技術調研報告

**日期**: 2026-04-05
**目的**: 為 CV-based notehead detector 尋找改進技術
**當前狀態**: morphology staff removal + connected component → 54-85% 覆蓋率（平均 64%）

---

## 1. Staff Line Removal 演算法

### 1.1 核心論文

| 論文 | 年份 | 方法 | 連結 |
|------|------|------|------|
| **Dalitz et al. "A comparative study of staff removal algorithms"** | 2008 | 比較 Skeleton, Run-length, Roach-Tatem, Carter, Fujinaga, Line-tracking | [Springer](https://link.springer.com/article/10.1007/s10032-009-0100-1) |
| **Cardoso & Capela "Staff Detection with Stable Paths"** | 2008 | Stable path + 圖論方法追蹤 staff lines | [SemanticScholar](https://www.semanticscholar.org/paper/Staff-Detection-with-Stable-Paths-Cardoso-Capela/c730cc618fb7490095afdd52e7de0091062e525b) / [PDF](https://www.inescporto.pt/~jsc/publications/conferences/2008ACapelaSIGMAP.pdf) |
| **Calvo-Zaragoza et al. "Staff-line detection and removal using CNN"** | 2017 | 用 CNN 學習哪些 pixel 是 staff line | [Springer](https://link.springer.com/article/10.1007/s00138-017-0844-4) |
| **Calvo-Zaragoza "Staff-line removal with Selectional Auto-Encoders"** | 2017 | SAE 自動學習 staff removal | [ResearchGate](https://www.researchgate.net/publication/318729877_Staff-line_removal_with_Selectional_Auto-Encoders) |
| **"Staff Detection and Removal" (book chapter)** | 2017 | 綜述所有方法 | [ResearchGate](https://www.researchgate.net/publication/314519129_Staff_Detection_and_Removal) |
| **"A morphological method for music score staff removal"** | 2015 | 形態學方法 | [ResearchGate](https://www.researchgate.net/publication/283129569_A_morphological_method_for_music_score_staff_removal) |
| **"Staff Line Detection and Removal in the Grayscale Domain"** | 2014 | 灰階域操作（不需二值化） | [ResearchGate](https://www.researchgate.net/publication/261096214_Staff_Line_Detection_and_Removal_in_the_Grayscale_Domain) / [Academia](https://www.academia.edu/4504624/Staff_line_Detection_and_Removal_in_the_Grayscale_Domain) |
| **"Staff detection and removal using derivation and connected component analysis"** | 2014 | 微分 + 連通元件 | [ResearchGate](https://www.researchgate.net/publication/261451292_Staff_detection_and_removal_using_derivation_and_connected_component_analysis) |
| **"A connected path approach for staff detection"** | 2009 | 連通路徑方法 | [ResearchGate](https://www.researchgate.net/publication/221127148_A_connected_path_approach_for_staff_detection_on_a_music_score) |

### 1.2 方法比較

| 方法 | 優點 | 缺點 | 適用性 |
|------|------|------|--------|
| **Horizontal morphology** (我們目前用的) | 簡單、快速 | 破壞 notehead-staff 交叉處 | ⚠️ 覆蓋率上限 ~65% |
| **Run-length encoding** | 保留 notehead-staff 交叉 | 保留 stems/beams 連接 | ⚠️ 需配合後處理 |
| **Stable paths (Cardoso)** | 精確追蹤每條 staff line | 實作複雜 | ✅ 理論最佳 |
| **CNN-based** | 最精確 | 需要訓練數據 | ✅ 但需額外訓練 |
| **Grayscale domain** | 不需二值化，保留更多資訊 | 參數敏感 | ✅ 可嘗試 |

### 1.3 OpenCV 官方教學

- [Extract horizontal and vertical lines by using morphological operations](https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html)
- [How to remove line on music sheet (OpenCV Q&A)](https://answers.opencv.org/question/53293/how-to-remove-line-on-music-sheet/)

---

## 2. Notehead Detection 技術

### 2.1 Template Matching

| 資源 | 連結 |
|------|------|
| OpenCV 官方教學 | [Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html) |
| PyImageSearch 深度教學 | [cv2.matchTemplate](https://pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/) |
| Multi-template matching | [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/multi-template-matching-with-opencv/) |
| 實用指南 | [Object Detection Lite: Template Matching](https://data-ai.theodo.com/en/technical-blog/object-detection-template-matching) |

**適用性**: LilyPond 生成的 noteheads 在同一字體下大小一致，非常適合 template matching。可從 DoReMi 圖片中裁切 notehead 模板。

### 2.2 Hough Circle/Ellipse Detection

| 資源 | 連結 |
|------|------|
| scikit-image Hough Ellipse | [文檔](https://scikit-image.org/docs/stable/auto_examples/edges/plot_circular_elliptical_hough_transform.html) |
| Randomized Hough Transform | [Wikipedia](https://en.wikipedia.org/wiki/Randomized_Hough_transform) |

**適用性**: Filled noteheads 近似圓形，Hough circle 可能有效。但計算量大，且需要先去除 staff lines。

### 2.3 Connected Component Analysis (我們目前的方法)

我們的方法基於 connected component + shape filters。主要參考：
- Blob area, circularity, solidity, aspect ratio
- 這是最常用的 OMR notehead detection 基礎方法

---

## 3. 開源 OMR 工具

### 3.1 完整 OMR 系統

| 專案 | 語言 | 方法 | 連結 | 備註 |
|------|------|------|------|------|
| **oemer** (End-to-end OMR) | Python | DL-based | [GitHub (BreezeWhite)](https://github.com/BreezeWhite/oemer) / [GitHub (meteo-team)](https://github.com/meteo-team/oemer) | ⭐ 最完整的開源 OMR |
| **cadenCV** (MIT 課程) | Python/OpenCV | Classical CV | [GitHub](https://github.com/afikanyati/cadenCV) | ⭐ 包含 staff removal + notehead detection |
| **overscore** | Python | Classical | [GitHub](https://github.com/acieroid/overscore) | Scheme + Python |
| **qandaomr** | Python | Hybrid | [GitHub](https://github.com/Jojojoppe/qandaomr) | |
| **rbaron/omr** | Python | Classical | [GitHub](https://github.com/rbaron/omr) | 簡潔實作 |
| **marcielbp/omr-opencv-python** | Python/OpenCV | Classical | [GitHub](https://github.com/marcielbp/omr-opencv-python) | |
| **dmgonzalez8/OMR** | Python | Classical | [GitHub](https://github.com/dmgonzalez8/OMR) | |
| **CUDA-OMR** | CUDA/C++ | GPU 加速 | [GitHub](https://github.com/MalekItani/CUDA-OMR) | GPU 加速 staff removal |
| **brandon-mason/opencv-sheet-music-player** | Python | OpenCV | [GitHub](https://github.com/brandon-mason/opencv-sheet-music-player) | |

### 3.2 Staff Line Removal 專項

| 專案 | 連結 |
|------|------|
| GitHub Topic: staff-line-removal | [Browse](https://github.com/topics/staff-line-removal) |
| Gamera MusicStaves Toolkit | [Site](https://gamera.informatik.hsnr.de/addons/musicstaves/) |

### 3.3 OMR 數據集工具

| 專案 | 連結 |
|------|------|
| **apacha/OMR-Datasets** | [GitHub](https://github.com/apacha/OMR-Datasets) / [文檔](https://apacha.github.io/OMR-Datasets/) |
| **MusicObjectDetector-TorchVision** | [GitHub](https://github.com/apacha/MusicObjectDetector-TorchVision) |
| **MusicObjectDetector-TF** | [GitHub](https://github.com/apacha/MusicObjectDetector-TF) |
| **MeasureDetector** | [GitHub](https://github.com/OMR-Research/MeasureDetector) |
| **Mensural-Detector** | [GitHub](https://github.com/apacha/Mensural-Detector) |
| **MusicSymbolClassifier** | [GitHub](https://github.com/apacha/MusicSymbolClassifier) |
| **MusicScoreClassifier** | [GitHub](https://github.com/apacha/MusicScoreClassifier) |
| **MusicObjectDetection** | [GitHub](https://github.com/apacha/MusicObjectDetection) |

---

## 4. 影像前處理技術

### 4.1 二值化方法

| 方法 | 適用場景 | 連結 |
|------|---------|------|
| **Adaptive Threshold** (我們目前用的) | 乾淨圖片 | OpenCV 內建 |
| **Sauvola Binarization** | 不均勻光照、掃描件 | `skimage.filters.threshold_sauvola` |
| **CLAHE** (我們已加入) | 低對比區域增強 | OpenCV `createCLAHE` |
| **Morphological Gradient** | 邊緣增強，跨字體一致性 | `cv2.MORPH_GRADIENT` |

### 4.2 大規模批次處理

| 方法 | 連結 |
|------|------|
| Python multiprocessing + OpenCV | 標準方案 |
| OpenCV GPU (CUDA) 加速 | [OpenCV Q&A](https://answers.opencv.org/question/233104/parallelizing-gpu-processing-of-multiple-images/) |
| GPU 加速影像處理 | [DeepWiki](https://deepwiki.com/opencv/opencv/14.2-gpu-accelerated-image-processing-and-optical-flow) |
| Parallel Batch Processing (GitHub) | [GitHub](https://github.com/Dharanya2005/Parallel-Batch-Image-Processing-System-for-Large-Scale-Datasets-using-Python-and-OpenCV) |

**我們的情況**：2,600 張圖 × ~50ms/張 ≈ 2 分鐘。不需要 GPU 加速。`multiprocessing.Pool` 即可。

---

## 5. Hybrid CV + DL 方法

### 5.1 學術論文

| 論文 | 年份 | 方法 | 連結 |
|------|------|------|------|
| **"State-of-the-Art Model for Music Object Recognition with Deep Learning"** | 2019 | YOLO/SSD for music symbols | [MDPI](https://www.mdpi.com/2076-3417/9/13/2645) |
| **"A Baseline for General Music Object Detection with Deep Learning"** (Pacha et al.) | 2018 | Faster R-CNN, RetinaNet 基線 | [MDPI](https://www.mdpi.com/2076-3417/8/9/1488) / [arXiv](https://arxiv.org/pdf/1801.07141) |
| **"Towards Full-Pipeline Handwritten OMR with Musical Symbol Detection by U-Nets"** | 2018 | U-Net pixel segmentation | [SemanticScholar](https://www.semanticscholar.org/paper/Towards-Full-Pipeline-Handwritten-OMR-with-Musical-Hajic-Dorfer/f05cb3674df33d35e562ca79b9b3af2e10c1a88a) / [DBLP](https://dblp.org/rec/conf/ismir/HajicDWP18.html) |
| **"Towards Self-Learning Optical Music Recognition"** | 2018 | Self-learning / 自訓練 | [ResearchGate](https://www.researchgate.net/publication/322667514_Towards_Self-Learning_Optical_Music_Recognition) |
| **"Introduction to Optical Music Recognition: Overview and Practical Challenges"** | 2021 | OMR 綜述 | [ACM](https://dl.acm.org/doi/fullHtml/10.1145/2181796.2206309) |
| **Sheet Music Transformer++** | 2024 | 端到端全頁 OMR | [arXiv](https://arxiv.org/html/2405.12105v4) |
| **"Comparative Analysis of Object Detection Models for Sheet Music Recognition: YOLO and OMR"** | 2024 | YOLO vs 傳統 OMR | [ResearchGate](https://www.researchgate.net/publication/385904852_Comparative_Analysis_of_Object_Detection_Models_for_Sheet_Music_Recognition_A_Focus_on_YOLO_and_OMR_Technologies) |
| **Calvo-Zaragoza "Staff-line removal"** (2017) | 2017 | CNN for staff pixel classification | [arXiv](https://arxiv.org/pdf/1906.08819) |
| **"Avoiding staff removal stage in OMR"** | 2014 | 跳過 staff removal 直接辨識 | [ResearchGate](https://www.researchgate.net/publication/268726980_Avoiding_staff_removal_stage_in_optical_music_recognition_application_to_scores_written_in_white_mensural_notation) |

### 5.2 OMR 綜述/入門

| 資源 | 連結 |
|------|------|
| Wikipedia: Optical music recognition | [Link](https://en.wikipedia.org/wiki/Optical_music_recognition) |
| OMR State-of-the-art and Open Issues (2012) | [ResearchGate](https://www.researchgate.net/publication/257806547_Optical_music_recognition_State-of-the-art_and_open_issues) |
| EmergentMind: OMR Topics | [Link](https://www.emergentmind.com/topics/optical-music-recognition-omr) |
| OMR Metrics and Evaluation: Systematic Review (2023) | [Springer](https://link.springer.com/article/10.1007/s13735-023-00278-5) |
| 最新 OMR Survey (2025) | [Springer](https://link.springer.com/article/10.1007/s13735-025-00385-5) |

### 5.3 實用教學

| 資源 | 連結 |
|------|------|
| YOLO + OMR 比較 (Medium) | [Article](https://medium.com/@sheneric211/a-tale-of-two-techniques-comparing-deep-learning-optical-music-recognition-with-ultralytics-yolov8-34b40f2d3bdf) |
| Sheet Music with Python + OpenCV | [Heartbeat](https://heartbeat.fritz.ai/play-sheet-music-with-python-opencv-and-an-optical-music-recognition-model-a55a3bea8fe) |
| CS766 OMR Project (Wisconsin) | [Page](https://pages.cs.wisc.edu/~vinitskys/omr/cs766_final.html) |

---

## 6. cadenCV / oemer 深度分析（2026-04-05 data-curator agent）

### cadenCV (MIT 課程)
- Staff removal: **不用 morphology**。RLE 偵測 staff row 後直接把 row±1 設為白色 — **比我們更粗暴**
- Notehead detection: **多尺度 template matching** (`cv2.matchTemplate` + `TM_CCOEFF_NORMED`)
- **結論**: 沒有可借鑒的 staff removal 技巧

### oemer (端到端 OMR)
- Staff removal: **完全 DL-based** (UNet 語義分割)，不適用（需要訓練數據）
- Notehead detection: 在 DL 產出的 notehead mask 上用 morphology 分離 + CC
- **可借鑒**: `unit_size`（staff spacing）作為所有尺寸參數的基準

### 最高價值改進：Conditional Staff Pixel Removal

```python
# 只移除「純 staff line」pixel（上下都沒有黑色鄰居的）
# 保留 notehead-staff 交叉處的 pixel
for each pixel (y, x) in staff_lines:
    if binary[y-1, x] == 0 or binary[y+1, x] == 0:
        no_staff[y, x] = binary[y, x]  # 保留（有垂直鄰居 = 符號一部分）
    else:
        no_staff[y, x] = 0  # 移除（純水平 = staff line）
```

---

## 7. 關鍵學術發現（2026-04-05 research agent）

### Dalitz 2008 比較研究結論
- **乾淨電腦生成的樂譜**（如 LilyPond）：所有方法表現都好
- Morphological opening = run-length 的等價形態學表達 → **我們的方法是正確的**
- 差異主要在退化/手寫樂譜

### ICDAR 2013 Staff Removal 冠軍
- Geraud 2014: Permissive hit-or-miss + horizontal median filter + reconstruction
- 也是 morphology-based → 再次驗證我們的方向

### 重要論文發現
| 論文 | 關鍵啟示 |
|------|---------|
| **arXiv:2409.00316** "Toward a More Complete OMR" | **在不完美偵測上訓練反而優於完美 GT** — 我們的 noisy pseudo-labels 可能就是對的 |
| **DART Pipeline** (arXiv:2407.09174) | 用 confidence thresholding 過濾 pseudo-labels 可提升品質 |
| **Moonlight** (Google) | 在 staff line 位置做 1D slice 分類，完全避開 staff removal 問題 |

### 額外開源工具
| 工具 | 方法 | 連結 |
|------|------|------|
| **homr** | oemer 改良 + Transformer | [GitHub](https://github.com/liebharc/homr) |
| **Audiveris** | Template matching for noteheads | [GitHub](https://github.com/Audiveris/audiveris) |
| **Orchestra** | Morphological staff removal | [GitHub](https://github.com/AbdallahHemdan/Orchestra) |
| **Moonlight** (Google) | 1D CNN on staff slices | [GitHub](https://github.com/tensorflow/moonlight) |
| **OMReader** | SIFT + decision tree | [GitHub](https://github.com/Etshawy1/OMReader) |

---

## 8. 結論與建議

### 當前狀態 (2026-04-05)
- CV notehead detector: 54-85% 覆蓋率（平均 64%），位置精準
- V2 pseudo-label dataset: 332K CV noteheads + 470K Phase 5 other classes
- 34/34 TDD tests passed

### 優先改進（如果需要更高覆蓋率）

1. **⭐ Conditional staff pixel removal** — 只移除無垂直鄰居的 staff pixel（data-curator 建議）
2. **Template matching 第二輪** — 用 LilyPond notehead 模板補充漏偵測
3. **unit_size-based filtering** — 用 staff spacing 動態調整 blob 大小閾值
4. **Confidence filtering** — 剔除低品質 pseudo-labels（DART 方法）

### 不需要做的
- GPU 加速（2600 張圖 CPU 2-3 分鐘就夠）
- DL-based staff removal（LilyPond 乾淨圖用 morphology 就好）
- Run-length staff removal（Dalitz 確認 morphology = run-length 等價）

---

*Updated: 2026-04-05, integrated findings from 3 research agents*
*Sources: 40+ papers, 15+ open-source repos, Dalitz 2008, ICDAR 2013, oemer, cadenCV*
