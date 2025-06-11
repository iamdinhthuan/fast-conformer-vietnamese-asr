# Phiên bản Fast Conformer

> **Nhận dạng giọng nói streaming chỉ với ~300 dòng mã.**

Kho lưu trữ này cung cấp một pipeline huấn luyện và suy luận ASR (Automatic Speech Recognition) gọn nhẹ nhưng sẵn sàng cho sản xuất, xây dựng trên **PyTorch Lightning** và **torchaudio ≥ 2.2**.  
Mô hình sử dụng **Fast Conformer encoder** (đã loại bỏ mọi biến thể Conformer cũ) và mục tiêu huấn luyện kết hợp **CTC + RNNT** với tùy chọn CTC phụ cho các tầng trung gian.

## ✨ Điểm nổi bật

• **Chỉ Fast Conformer** – thân thiện streaming, tuỳ chỉnh ngữ cảnh trái/phải.  
• **Loss** – CTC, RNNT và CTC trung gian giúp hội tụ nhanh hơn.  
• **Checkpoint theo bước** – lưu checkpoint **WER tốt nhất** + checkpoint FP16 (chỉ weights) định kỳ (~200 MB).  
• **Điều khiển bằng config** – mọi siêu tham số nằm trong `config.py` / `config.json`.  
• **Suy luận streaming** với cửa sổ 640 ms, chồng lấn 160 ms.  
• **Phụ thuộc tối thiểu** – thuần PyTorch, không cần Fairseq/SentencePiece.

---

## 📂 Cấu trúc thư mục

```text
├── config.py               # Dataclass chứa mặc định
├── config.json             # Ví dụ config (ghi đè mặc định)
├── rnnt_lightning.py       # LightningModule cho huấn luyện RNN-T only
├── run.py                  # Điểm vào huấn luyện (argparse + Lightning Trainer)
├── inference.py            # Suy luận offline
├── streaming_inference.py  # Demo suy luận thời gian thực
├── models/
│   ├── fast_conformer.py   # Fast Conformer encoder (torchaudio ≥ 2.2)
│   ├── rnnt_decoder.py     # RNN-T decoder với LSTMCell (Lightning compatible)
│   └── rnnt_streaming.py   # Streaming RNN-T decoder cho real-time inference
└── utils/                  # Tải dữ liệu, augmentation, metrics, ...
```

---

## ⚡ Bắt đầu nhanh

1. **Cài đặt phụ thuộc** (khuyên dùng Python ≥ 3.9):

   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   # torchaudio phải khớp phiên bản CUDA/CPU của PyTorch.
   ```

2. **Chuẩn bị dữ liệu**: tạo file `metadata.csv` theo định dạng

   ```text
   /duong_dan/tuyet_doi/audio_0001.wav|xin chào thế giới
   /duong_dan/tuyet_doi/audio_0002.wav|general kenobi
   ```

   Cập nhật lại đường dẫn trong `config.json → data` cho phù hợp.

3. **Huấn luyện**:

   ```bash
   python run.py --config config.json
   ```

4. **Validation Prediction** (tùy chọn):

   Trong quá trình training, model sẽ tự động predict:
   - 5 samples ngẫu nhiên từ validation set
   - Tất cả file audio trong thư mục tùy chỉnh (nếu có)

   ```bash
   # Chỉ định thư mục để predict thêm
   python run.py --config config.json --val-predict-dir ./test_audio

   # Tùy chỉnh số lượng random samples
   python run.py --config config.json --val-predict-samples 10
   ```

   Một số tuỳ chọn ghi đè nhanh:

   ```bash
   python run.py --config config.json --batch-size 16 --learning-rate 2e-4
   # Tiếp tục từ checkpoint
   python run.py --config config.json --resume checkpoints/last.ckpt
   ```

4. **Suy luận offline**:

   ```bash
   python inference.py --wav test.wav --checkpoint checkpoints/best-wer.ckpt
   ```

5. **Suy luận streaming** (độ trễ thấp):

   ```bash
   python streaming_inference.py \
       --wav test.wav \
       --checkpoint weights/encoder_ctc.pt \
       --left-ctx 160 --right-ctx 40
   ```

---

## 🛠️ Bảng tham chiếu cấu hình (`config.py` / `config.json`)

| Nhóm         | Khoá                          | Ý nghĩa                                            |
|--------------|------------------------------|----------------------------------------------------|
| `model`      | `n_state` / `n_head` / `n_layer` | Chiều và độ sâu Transformer                       |
|              | `left_ctx` / `right_ctx`     | Ngữ cảnh streaming (frame)                         |
|              | `dropout` / `ffn_expansion`  | Regularization & scale FFN                         |
| `training`   | `batch_size` / `learning_rate` | Tham số tối ưu hoá                                |
|              | `checkpoint_every_n_steps`   | Khoảng lưu checkpoint FP16 (chỉ weights)           |
|              | `val_check_interval`         | Tần suất validate (tính theo step)                 |
| `loss`       | `lambda_ctc` / `aux_loss_weight` | Tỷ lệ pha trộn RNNT–CTC & CTC trung gian        |
| `paths`      | `data_dir` / `log_dir` / `checkpoint_dir` | Thư mục dữ liệu, log, checkpoint      |

Mọi trường đã có giá trị mặc định trong `config.py`; `config.json` chỉ cần ghi đè khi bạn muốn thay đổi.

---

## 📝 Lưu ý huấn luyện

1. **Phần cứng** – thử nghiệm trên 1 GPU NVIDIA (≥ 12 GB).  
2. **Mixed precision** – bật sẵn (`precision=16-mixed`).  
3. **Gradient accumulation** – dùng `training.accumulate_grad_batches` để nhồi batch lớn.  
4. **Augmentation** – `utils/dataset.py` hỗ trợ SpecAugment, time-stretch, trộn nhiễu; tắt bằng `augment=False`.

---

## 🤝 Cảm ơn

Dự án tham khảo và tái sử dụng ý tưởng từ:  
• [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) – Fast Conformer & RNNT loss  
• [torchaudio](https://github.com/pytorch/audio) – Lớp encoder & `RNNTLoss`  
• [OpenAI Whisper](https://github.com/openai/whisper) – Tokeniser & mẹo decoding

---

## 📄 Giấy phép

Phát hành theo giấy phép **Apache 2.0** – chi tiết xem file `LICENSE`.
