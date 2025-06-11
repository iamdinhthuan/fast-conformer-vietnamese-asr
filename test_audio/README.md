# Test Audio Directory

Đặt các file audio test vào thư mục này để model predict trong quá trình validation.

## Supported formats:
- .wav
- .mp3
- .flac
- .m4a
- .ogg

## Usage:

### 1. Sử dụng config.json:
```json
{
  "training": {
    "val_predict_dir": "./test_audio"
  }
}
```

### 2. Sử dụng command line:
```bash
python run.py --config config.json --val-predict-dir ./test_audio
```

### 3. Tùy chỉnh số lượng random samples:
```bash
python run.py --config config.json --val-predict-samples 10
```

## Kết quả:

Mỗi khi validation kết thúc, model sẽ:
1. Predict 5 samples ngẫu nhiên từ validation set
2. Predict tất cả file audio trong thư mục này (tối đa 10 files)
3. Log kết quả prediction và WER (nếu có ground truth)

## Ví dụ output:
```
🎯 Predicting 5 random validation samples...
📝 Random Sample 1/5 (idx=123):
   Target: 'xin chào tôi là trợ lý ảo'
   Prediction: 'xin chào tôi là trợ lý ảo'
   WER: 0.0000

🎯 Predicting 3 files from ./test_audio...
📁 File 1/3: test_sample.wav
   Prediction: 'đây là file test audio'
   Duration: 2.34s
```
