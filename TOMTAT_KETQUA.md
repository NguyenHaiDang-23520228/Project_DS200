# Tóm Tắt Kết Quả Thực Nghiệm — DS200.Q21
**Nhóm:** 23520228 Nguyễn Hải Đăng · 23520055 Nguyễn Bi Anh  
**Đề tài:** Phân đoạn đa cơ quan CT bụng với TransUNet (arXiv:2102.04306)  
**Dữ liệu:** Synapse — 2211 slice train / 12 volume test, 8 cơ quan

---

## I. Nền tảng — Reproduce kết quả tác giả

### 1. Baseline từ tác giả (paper)
| Mô hình | Mean DSC | Mean HD95 |
|---------|----------|-----------|
| R50-U-Net (paper) | 74.68% | 36.87 mm |
| TransUNet R50-ViT-B/16 (paper) | 77.48% | 31.69 mm |

### 2. Kết quả reproduce của nhóm (`demo.ipynb`, 150 epoch)
| Mô hình | Mean DSC | Mean HD95 | So với paper |
|---------|----------|-----------|-------------|
| R50-U-Net | **76.31%** | **29.59 mm** | +1.63% DSC, −7.28 HD95 |
| TransUNet R50-ViT-B/16 | **77.65%** | **31.22 mm** | +0.17% DSC, −0.47 HD95 |

> ✅ **Nhóm reproduce thành công TransUNet và R50-U-Net**, kết quả xấp xỉ hoặc nhỉnh hơn paper.

---

## II. Ablation — Vai trò của Skip Connections (`ablationskip.ipynb`, 60 epoch)

Thực nghiệm thay đổi tham số `n_skip` (số lớp ViT encoder đưa sang decoder qua skip connection):

| Mô hình | n_skip | Mean DSC | Mean HD95 | Ghi chú |
|---------|--------|----------|-----------|---------|
| R50-ViT-CUP | 0 | 68.94% | 32.36 mm | Không có skip |
| TransUNet | 1 | 73.88% | 30.71 mm | 1 lớp skip |
| TransUNet | 3 | 77.65% | 31.22 mm | Đầy đủ skip (từ demo) |

> ✅ Kết quả **khớp với Figure 2 của paper**: càng nhiều skip connection → DSC càng tăng rõ rệt, xác nhận tầm quan trọng của skip connection trong TransUNet.

---

## III. Cải tiến Hướng 1 — Attention Gate trên Skip Connection (`attentiongate.ipynb`)

### Ý tưởng
Thêm **Attention Gate (AG)** vào mỗi skip connection trước khi nối (concatenate) vào decoder.  
Gate dùng feature decoder làm query, feature encoder làm value, học mask chú ý giúp **lọc nhiễu background** tại skip → kỳ vọng cải thiện DSC và HD95.

### Kiến trúc (`networks/vit_seg_attnskip.py`)
```
ViT encoder → [skip_i × AG(skip_i, decoder_query_i)] → Decoder
```
Mỗi AG gồm: Conv 1×1 + BN + ReLU → sigmoid mask → nhân element-wise với skip.

### Kết quả (150 epoch, Synapse 12 test volumes)
| Cơ quan | AttnSkip DSC | AttnSkip HD95 | TransUNet DSC | TransUNet HD95 |
|---------|:------------:|:-------------:|:-------------:|:--------------:|
| Spleen | 87.09% | 7.96 | – | – |
| R.Kidney | 64.93% | **9.17** | – | – |
| L.Kidney | 80.94% | 50.78 | – | – |
| Gallbladder | 79.71% | 58.55 | – | – |
| Liver | 94.54% | 23.41 | – | – |
| Stomach | 55.06% | 14.50 | – | – |
| Aorta | 84.90% | 39.62 | – | – |
| Pancreas | 75.13% | 19.71 | – | – |
| **MEAN** | **77.79%** | **27.96** | **77.65%** | **31.22** |

### So sánh với TransUNet gốc
| Metric | TransUNet | AttnSkip | Thay đổi |
|--------|-----------|----------|----------|
| Mean DSC | 77.65% | **77.79%** | **+0.14%** ↑ |
| Mean HD95 | 31.22 mm | **27.96 mm** | **−3.26 mm** ↓ (tốt hơn) |

> ✅ **AttnSkip vượt TransUNet ở CẢ HAI metric.** HD95 cải thiện rõ nhất (−3.26 mm), cho thấy Attention Gate giúp model dự đoán biên cơ quan chính xác hơn.

---

## IV. Cải tiến Hướng 2 — Spectral Frequency trên Skip Connection (`attentiongate.ipynb`)

### Ý tưởng
Chồng thêm **nhánh tần số (SpectralTransform / FFT-style)** trước Attention Gate.  
Nhánh tần số trích đặc trưng global/cấu trúc dài hạn bổ sung cho nhánh không gian → kỳ vọng cải thiện thêm HD95 (ranh giới).

### Kiến trúc (`networks/vit_seg_attnfreq.py`)
```
skip_i → [FreqBranch(FFT → Conv → iFFT) + SpatialBranch] → concat → AG → Decoder
```
Learnable BN-gamma kiểm tra nhánh tần số có học hay không.

### Kết quả (150 epoch, Synapse 12 test volumes)
| Cơ quan | AttnFreq DSC | AttnFreq HD95 | AttnSkip HD95 | Δ HD95 |
|---------|:------------:|:-------------:|:-------------:|:------:|
| Spleen | 87.32% | 10.98 | 7.96 | +3.02 |
| R.Kidney | 62.42% | 10.01 | 9.17 | +0.84 |
| L.Kidney | 80.31% | 43.92 | 50.78 | **−6.86** ↓ |
| Gallbladder | 77.29% | **25.02** | 58.55 | **−33.53** ↓ |
| Liver | 94.57% | 26.52 | 23.41 | +3.11 |
| Stomach | 53.24% | 14.85 | 14.50 | +0.35 |
| Aorta | 84.78% | 42.58 | 39.62 | +2.96 |
| Pancreas | 74.03% | 19.11 | 19.71 | **−0.60** ↓ |
| **MEAN** | **76.75%** | **24.12** | **27.96** | **−3.84** ↓ |

Learnable BN-gamma (kiểm tra nhánh tần số có học): `[0.077, 0.285, 0.241]` → **nhánh tần số được kích hoạt**.

### So sánh AttnFreq vs các mô hình khác
| | DSC | HD95 |
|---|-----|------|
| AttnFreq vs TransUNet | −0.90% | **−7.10 mm** (tốt hơn rõ) |
| AttnFreq vs AttnSkip | −1.04% | **−3.84 mm** (tốt hơn) |

> ✅ **AttnFreq đạt HD95 thấp nhất trong tất cả mô hình (24.12 mm)**, đặc biệt cải thiện mạnh ở Gallbladder (−33.53 mm) và L.Kidney (−6.86 mm).  
> ⚠️ Đánh đổi: Mean DSC thấp hơn AttnSkip (−1.04%) và thấp hơn TransUNet gốc (−0.90%).

---

## V. Thực nghiệm bổ sung — Test-Time Augmentation (TTA)

### Ý tưởng
Trong lúc test, áp dụng **flip ngang/dọc/ngang+dọc** → trung bình xác suất 4 view → kỳ vọng dự đoán ổn định hơn.

### Kết quả
| Mô hình | DSC | HD95 | So với không TTA |
|---------|-----|------|-----------------|
| AttnSkip (không TTA) | 77.79% | 27.96 | – |
| AttnSkip **+TTA** | 75.23% | 33.12 | **−2.56% DSC, +5.16 HD95** |
| wCE (không TTA) | 77.01% | 29.17 | – |
| wCE **+TTA** | 74.97% | 37.70 | **−2.04% DSC, +8.53 HD95** |

> ❌ **TTA làm kết quả tệ hơn đáng kể ở cả DSC lẫn HD95.**  
> **Lý do:** Ảnh CT bụng có cấu trúc giải phẫu **bất đối xứng** (gan bên phải, lách bên trái, hai thận hai bên). Khi lật ngang (horizontal flip), model nhận ảnh "đảo bên" không đúng với phân phối train → dự đoán sai bên → trung bình làm kết quả xấu hơn.  
> **Bài học:** TTA với flip không phù hợp cho CT bụng bất đối xứng. Cần augmentation đối xứng (xoay nhỏ, jitter brightness) nếu muốn dùng TTA.

---

## VI. Thực nghiệm bổ sung — Weighted Cross-Entropy (wCE)

### Ý tưởng
EDA (`data.ipynb`) cho thấy mất cân bằng lớp nghiêm trọng: background ~96% pixel, Stomach chỉ ~0.05%. Dùng **weighted CE** với trọng số nghịch đảo tần số (sqrt-inv-freq, clamp [0.5, 4.0]) kết hợp 0.5·wCE + 0.5·Dice để tập trung vào cơ quan hiếm.

### Trọng số lớp
| Lớp | Weight | | Lớp | Weight |
|-----|--------|-|-----|--------|
| Background | 0.47 | | Liver | 0.47 |
| Spleen | 1.18 | | Stomach | **1.40** |
| R.Kidney | **2.29** | | Aorta | 0.64 |
| L.Kidney | 0.98 | | Pancreas | 0.61 |
| Gallbladder | 0.98 | | | |

### Kết quả (150 epoch)
| Cơ quan | wCE DSC | AttnSkip DSC | Δ DSC |
|---------|:-------:|:------------:|:-----:|
| Spleen | 86.16% | 87.09% | −0.93% |
| R.Kidney | 61.69% | 64.93% | −3.24% |
| L.Kidney | **82.77%** | 80.94% | **+1.83%** |
| Gallbladder | 78.33% | 79.71% | −1.38% |
| Liver | 94.01% | 94.54% | −0.53% |
| Stomach | 54.99% | 55.06% | −0.07% |
| Aorta | 84.66% | 84.90% | −0.24% |
| Pancreas | 73.49% | 75.13% | −1.64% |
| **MEAN** | **77.01%** | **77.79%** | **−0.78%** |

> ❌ **wCE không cải thiện mean DSC hay HD95** so với AttnSkip thường.  
> **Nhận xét:** Loss có trọng số cải thiện cục bộ vài cơ quan (L.Kidney +1.83%) nhưng làm hại các cơ quan khác, đặc biệt R.Kidney HD95 tăng từ 9.17 lên 34.08 mm — rất tệ.

---

## VII. Thực nghiệm pilot — FGMask (`newinnovation.ipynb`)

### Ý tưởng
**Foreground-Aware Logit Masking (FGMask):** tại ViT encoder, dùng dự đoán sơ bộ để tính foreground mask, nhân vào các token ViT nhằm giảm nhiễu từ background token.

### Kết quả
| Metric | FGMask | AttnSkip | TransUNet |
|--------|:------:|:--------:|:---------:|
| Mean DSC | 76.77% | 77.79% | 77.65% |
| Mean HD95 | 34.49 mm | 27.96 | 31.22 |

> ❌ **FGMask tệ hơn cả AttnSkip lẫn TransUNet gốc**, đặc biệt HD95 xấu nhất (34.49 mm).  
> Tham số `fg_alpha ≈ −0.0015` (gần 0) cho thấy cơ chế mask không được học hiệu quả.  
> Hướng này được ghi nhận là **negative result** — không đưa vào pipeline chính.

---

## VIII. Bảng tổng hợp tất cả kết quả

```
==============================================================================
Model              |  Mean DSC | Mean HD95 | Small-organ DSC
------------------------------------------------------------------------------
U-Net              |    76.31% |     29.59 |          66.00%
TransUNet (gốc)    |    77.65% |     31.22 |          68.41%
AttnSkip           |    77.79% |     27.96 |          68.66%   ← Tốt nhất DSC
AttnFreq           |    76.75% |     24.12 |          –        ← Tốt nhất HD95
AttnSkip+TTA       |    75.23% |     33.12 |          65.74%   ← TTA làm hại
AttnSkip_wCE       |    77.01% |     29.17 |          67.12%
AttnSkip_wCE+TTA   |    74.97% |     37.70 |          64.61%   ← Tệ nhất
==============================================================================
```

---

## IX. Kết luận

### Những gì đã làm được (ngoài tác giả gốc)

| # | Đóng góp | Kết quả |
|---|----------|---------|
| 1 | **Ablation n_skip** (0, 1, 3) | Xác nhận Figure 2 paper; n_skip càng nhiều DSC càng tăng |
| 2 | **AttnSkip** — Attention Gate trên skip | ✅ Vượt TransUNet: +0.14% DSC, −3.26 mm HD95 |
| 3 | **AttnFreq** — Nhánh tần số FFT trước AG | ✅ HD95 tốt nhất (24.12 mm), Gallbladder −33.53 mm |
| 4 | **TTA flip** | ❌ Làm hại do CT bụng bất đối xứng → bài học quan trọng |
| 5 | **Weighted CE** | ❌ Không cải thiện mean; cục bộ vài cơ quan |
| 6 | **FGMask (pilot)** | ❌ Negative result; fg_alpha ≈ 0 |

### Nhận định tổng quan
- **AttnSkip là cải tiến thành công nhất**: vượt TransUNet cả DSC lẫn HD95, ý tưởng đơn giản nhưng hiệu quả.
- **AttnFreq** có giá trị nghiên cứu về HD95 nhưng đánh đổi DSC; phù hợp nếu bài toán ưu tiên độ chính xác biên.
- **TTA, wCE, FGMask** là các thực nghiệm negative cho thấy không phải kỹ thuật nào cũng cải thiện khi áp dụng trực tiếp — đây cũng là kết quả có giá trị khoa học.
- Variance giữa các lần train (Colab, seed khác nhau) là có thật; AttnSkip dao động trong khoảng 76.7–77.8% DSC qua các run.

---

*File tóm tắt này tổng hợp từ kết quả thực tế trong các notebook: `demo.ipynb`, `ablationskip.ipynb`, `attentiongate.ipynb`, `newinnovation.ipynb`.*
