# DS200.Q21 — Phân đoạn đa cơ quan CT bụng với TransUNet

Dự án reproduce và mở rộng thí nghiệm từ bài báo **TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation** ([arXiv:2102.04306](https://arxiv.org/abs/2102.04306)) trên bộ dữ liệu **Synapse multi-organ**.

**Tác giả gốc:** Chen et al., 2021  
**Mục tiêu đồ án:** Reproduce TransUNet trên Synapse, so sánh với baseline CNN, ablation skip-connection theo paper, và thử một cải tiến (Attention Gate trên skip).

---

## Tóm tắt

TransUNet kết hợp **ResNet-50 + Vision Transformer (ViT-B/16)** làm encoder và **decoder kiểu U-Net** với skip-connection từ CNN. Paper chứng minh Transformer mạnh ở ngữ nghĩa toàn cục nhưng **cần skip U-Net** để giữ chi tiết không gian — ablation `n_skip = 0 → 1 → 3` cho thấy DSC tăng dần.

Dự án này:

1. **Reproduce** TransUNet (`n_skip=3`, 150 epoch) — sát số paper.
2. **Baseline** R50-U-Net trên cùng pipeline.
3. **Ablation** `n_skip ∈ {0, 1, 3}` — xác nhận xu hướng Figure 2 paper.
4. **Cải tiến** `TransUNet_AttnSkip` (Attention Gate trên 3 nhánh skip) — kết quả **không vượt** baseline gốc.

---

## Kiến trúc (theo paper)

```
Input CT (224×224)
    ↓
ResNet-50 (CNN) ──skip──┐
    ↓                   │
Hybrid ViT-B/16         │  n_skip ∈ {0,1,3}
    ↓                   │
Decoder (U-Net style) ←─┘
    ↓
Mask 9 lớp (8 cơ quan + background)
```

| Thành phần | Mô tả |
|------------|--------|
| Encoder | ResNet-50 + ViT-B/16 pretrained (ImageNet-21k) |
| Decoder | Upsampling + concat skip từ ResNet |
| `n_skip=0` | R50-ViT-CUP — không skip |
| `n_skip=1,3` | TransUNet — 1 hoặc 3 nhánh skip |
| Loss | Cross-entropy (theo repo gốc) |
| Metric | **Mean DSC (%)** và **Mean HD95 (mm)** trên 12 test volumes |

---

## Dữ liệu — Synapse Multi-Organ

| | Chi tiết |
|--|----------|
| Nguồn gốc | [Synapse multi-organ CT](https://www.synapse.org/#!Synapse:syn3193805/wiki/) |
| Preprocess | Theo pipeline TransUNet (2D slices train, 3D volumes test) |
| Train | **2.211** slice `.npz` (`train_npz/`) |
| Test | **12** volume `.npy.h5` (`test_vol_h5/`) |
| Kaggle (Colab) | [`dogcdt/synapse`](https://www.kaggle.com/datasets/dogcdt/synapse) |

### Ánh xạ nhãn (9 class)

| ID | Cơ quan |
|----|---------|
| 0 | Background |
| 1 | Spleen |
| 2 | Right kidney |
| 3 | Left kidney |
| 4 | Gallbladder |
| 5 | Liver |
| 6 | Stomach |
| 7 | Aorta |
| 8 | Pancreas |

Khảo sát dữ liệu và visualize: **`data.ipynb`**.

---

## Cấu trúc repository

```
DS200.Q21/
├── README.md                 # File này
├── data.ipynb                # EDA Synapse (local)
├── demo.ipynb                # Pipeline chính: TransUNet + U-Net + so sánh
├── ablationskip.ipynb              # Ablation n_skip = 0, 1 (vs 3 từ demo)
├── attentiongate.ipynb       # TransUNet_AttnSkip (Attention Gate)
├── data/
│   └── manifest.csv
└── project_TransUNet/
    ├── TransUNet/            # Code gốc / fork Beckschen/TransUNet
    │   ├── train.py, test.py
    │   ├── networks/
    │   └── datasets/
    └── data/Synapse/         # Dữ liệu local (nếu có)
        ├── train_npz/
        └── test_vol_h5/
```

---

## Notebooks — thứ tự chạy

| Notebook | Nội dung | Môi trường |
|----------|----------|------------|
| **`data.ipynb`** | Thống kê 2211 train / 12 test, label map, visualize slice | Local |
| **`demo.ipynb`** | Setup Colab → train TransUNet 150ep → train/test U-Net → bảng + overlay | **Google Colab (GPU)** |
| **`ablationskip.ipynb`** | Ablation `--n_skip 0` và `1` (60ep), bảng + biểu đồ vs `n_skip=3` | Colab |
| **`attentiongate.ipynb`** | `TransUNet_AttnSkip`, train 150ep, so với demo | Colab |

> **Lưu ý:** `demo.ipynb`, `ablationskip.ipynb`, `attentiongate.ipynb` được thiết kế cho Colab (`/content/...`). Chạy local cần chỉnh đường dẫn data và checkpoint.

---

## Kết quả chính

### So sánh mô hình (Synapse test, 12 volumes)

| Mô hình | Mean DSC (%) | Mean HD95 (mm) | Epoch | Ghi chú |
|---------|-------------|----------------|-------|---------|
| **TransUNet** (`n_skip=3`) | **77.65** | 31.22 | 150 | Reproduce — sát paper |
| Paper TransUNet | 77.48 | 31.69 | 150 | Chen et al., Table 1 |
| **R50-U-Net** | 76.31 | **29.59** | 150 | Baseline CNN |
| **TransUNet_AttnSkip** | 76.81 | 33.89 | 150 | Cải tiến — thua gốc |
| R50-ViT-CUP (`n_skip=0`) | 68.94 | 32.36 | 60 | Ablation |
| TransUNet (`n_skip=1`) | 73.88 | 30.71 | 60 | Ablation |
| Paper R50-ViT-CUP | 71.29 | 32.87 | 150 | Figure 2 paper |

**Nhận xét:**

- TransUNet reproduce **~77.65% DSC** — khớp paper (**77.48%**).
- U-Net có **HD95 tốt hơn** TransUNet (29.59 vs 31.22) dù DSC thấp hơn.
- Ablation skip: **DSC(0) < DSC(1) < DSC(3)** — đúng xu hướng paper; số 0/1 thấp hơn paper một phần do **60 vs 150 epoch**.
- Attention Gate trên 3 skip **không cải thiện** so với concat thuần (DSC −0.84%, HD95 +2.67).

---

## Cấu hình huấn luyện (mặc định)

| Tham số | Giá trị |
|---------|---------|
| `--vit_name` | `R50-ViT-B_16` |
| `--dataset` | `Synapse` |
| `--num_classes` | `9` |
| `--img_size` | `224` |
| `--batch_size` | `24` |
| `--base_lr` | `0.01` |
| `--max_epochs` | 150 (`demo`, `attentiongate`) / 60 (`huong3`) |
| `--n_skip` | `3` (TransUNet đầy đủ) |
| Pretrained ViT | [Google ViT ImageNet-21k](https://console.cloud.google.com/storage/vit_models/) |

---

## Chạy nhanh trên Google Colab

1. **Runtime → Change runtime type → GPU** (T4 / A100).
2. Mở **`demo.ipynb`** → **Run All** (cell đầu cài dependency, clone repo, tải data).
3. Cấu hình Kaggle API: đặt `kaggle.json` trong `~/.kaggle/` (không commit key lên Git).
4. Cell train TransUNet (~27 phút/epoch trên A100, batch 24).
5. Cell test + visualization.

Train/test thủ công (trong repo TransUNet):

```bash
# Train TransUNet (n_skip=3)
python train.py \
  --dataset Synapse \
  --vit_name R50-ViT-B_16 \
  --root_path /path/to/Synapse/train_npz \
  --list_dir ./lists/lists_Synapse \
  --num_classes 9 \
  --n_skip 3 \
  --max_epochs 150 \
  --batch_size 24 \
  --base_lr 0.01

# Test
python test.py \
  --dataset Synapse \
  --vit_name R50-ViT-B_16 \
  --volume_path /path/to/Synapse/test_vol_h5 \
  --list_dir ./lists/lists_Synapse \
  --num_classes 9 \
  --n_skip 3 \
  --max_epochs 150
```

Ablation `n_skip`: xem **`ablationskip.ipynb`** hoặc đổi `--n_skip 0` / `1`.

---

## Phụ thuộc

Theo [TransUNet/requirements.txt](project_TransUNet/TransUNet/requirements.txt):

- `torch`, `torchvision`, `numpy`, `h5py`, `scipy`
- `medpy`, `SimpleITK` (metric HD95)
- `ml-collections`, `tensorboardX`, `tqdm`

Colab notebooks cài thêm: `kaggle`, `matplotlib`.

---

## Kết luận đồ án

1. **Reproduce thành công** TransUNet trên Synapse — xác nhận Transformer + U-Net skip là phương án mạnh cho phân đoạn đa cơ quan.
2. **Ablation skip** tái hiện insight paper: bỏ skip làm DSC giảm mạnh (~69% vs ~78%).
3. **Attention Gate** trên skip (khác hướng paper Mục 4.4 — Transformer *trong* skip) **không beat** baseline; negative result có giá trị cho báo cáo.
4. **Hướng mở rộng:** train ablation 0/1-skip đủ 150 epoch; thử transformer trong skip như paper; so sánh literature (Swin-UNet) không bắt buộc train.

---

## Tài liệu tham khảo

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L. and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```

- Paper: [2102.04306v1.pdf](https://arxiv.org/pdf/2102.04306.pdf)
- Code gốc: [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet)
- Dataset: [Synapse multi-organ segmentation challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/)

---

## License & dữ liệu

- Code TransUNet: theo LICENSE trong `project_TransUNet/TransUNet/`.
- Dữ liệu Synapse và bản preprocess của TransUNet **chỉ dùng cho nghiên cứu**, không redistribute (xem `project_TransUNet/README.txt`).

---

## Tác giả

**DS200.Q21** 
— 23520228 - Nguyễn Hải Đăng
- 23520055 - Nguyễn Bi Anh
