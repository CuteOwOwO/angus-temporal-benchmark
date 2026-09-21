# 另一台機器：Audio Counting 模型下載與環境安裝手冊

版本日期：2026-09-21
用途：讓接手者在另一台 Linux/NVIDIA 機器上，從零下載並執行本專案使用或候選的 audio-language models。本文件不假設對方擁有原機器的 `/work/u4161854` 路徑。

## 1. 先決定要裝哪一組

### 建議先裝：已實際跑通的核心組

| Model ID | 用途 | checkpoint 約略空間 | 已觀察單卡峰值 VRAM |
|---|---|---:|---:|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Generator + Judge | 約 66 GB | 約 69 GiB |
| `openbmb/MiniCPM-o-4_5` | Judge | 約 18 GB | 約 22 GiB |
| `google/gemma-4-12B-it` | Judge | 約 23 GB | 約 24 GiB |
| `nvidia/audio-flamingo-3-hf` | Generator / Judge candidate | 約 33 GB | 約 33 GiB |

四個 checkpoint 約需 140 GB；加上 Hugging Face cache、Python environments、暫存及編譯空間，建議至少準備 **250 GB 可用磁碟**。如果只跑前三個，仍建議至少 200 GB。

### 完整候選組

再加入：

- `Qwen/Qwen3-Omni-30B-A3B-Thinking`
- `stepfun-ai/Step-Audio-2-mini`
- `stepfun-ai/Step-Audio-2-mini-Think`（已有格式失敗紀錄，目前 formal 排除）
- `moonshotai/Kimi-Audio-7B-Instruct`
- `google/gemma-3n-E4B-it`（gated）
- `microsoft/Phi-4-multimodal-instruct`
- `XiaomiMiMo/MiMo-Audio-7B-Instruct`
- `XiaomiMiMo/MiMo-Audio-Tokenizer`（MiMo 必要附屬模型）

完整組建議預留 **至少 450 GB**，最好 500 GB。不要把 Hugging Face cache 放在容量很小的 home partition。

## 2. 硬體與系統需求

- Linux x86_64。
- NVIDIA driver 可支援所選 PyTorch CUDA build。
- CUDA 12.x；本專案成功環境是 PyTorch `2.7.1+cu128` / CUDA runtime 12.8。
- 單卡 80 GB GPU 可從短音訊、batch size 1 開始；建議 H100 80 GB、H200 141 GB 或同級 GPU。24/48 GB 卡不適合直接以 BF16 跑完整 Qwen3-Omni。
- RAM 建議至少 128 GB；完整多模型節點建議 200 GB 以上。

系統套件：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg libsndfile1-dev sox build-essential ninja-build
git lfs install
```

若沒有 sudo，請管理員安裝上述套件。Kimi 官方優先建議 Docker；使用 Docker 時還需要 NVIDIA Container Toolkit。

## 3. 共用下載環境

以下以 `/data/audio_models` 為例，請換成對方的大容量磁碟。

```bash
export MODEL_ROOT=/data/audio_models
export HF_HOME=/data/huggingface
export HF_HUB_CACHE=/data/huggingface/hub
mkdir -p "$MODEL_ROOT" "$HF_HUB_CACHE"

python3.11 -m venv /data/venvs/hf-download
source /data/venvs/hf-download/bin/activate
python -m pip install -U pip "huggingface_hub[cli]"
hf auth login
```

- Token 只輸入互動式提示，不要寫進 shell script、Git、Slurm log 或文件。
- Gemma 3n 與其他受限 repo 要先在 Hugging Face 網頁接受 license/access terms。
- 正式實驗應 pin commit SHA，並保存 revision、config hash 與權重清單。
- `hf download --local-dir` 會把 checkpoint materialize 到指定目錄；不要依賴跨磁碟 cache symlink。

## 4. Checkpoint 下載命令

### 4.1 核心四模型

```bash
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
  --local-dir "$MODEL_ROOT/Qwen3-Omni-30B-A3B-Instruct"

hf download openbmb/MiniCPM-o-4_5 \
  --revision 073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1 \
  --local-dir "$MODEL_ROOT/MiniCPM-o-4_5"

hf download google/gemma-4-12B-it \
  --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --local-dir "$MODEL_ROOT/gemma-4-12B-it"

hf download nvidia/audio-flamingo-3-hf \
  --revision 7d4bae64ee29878af6504ae6f6bb3e40492838ad \
  --local-dir "$MODEL_ROOT/audio-flamingo-3-hf"
```

### 4.2 其餘候選模型

```bash
hf download Qwen/Qwen3-Omni-30B-A3B-Thinking \
  --revision 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b \
  --local-dir "$MODEL_ROOT/Qwen3-Omni-30B-A3B-Thinking"

hf download stepfun-ai/Step-Audio-2-mini \
  --revision e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36 \
  --local-dir "$MODEL_ROOT/Step-Audio-2-mini"

hf download stepfun-ai/Step-Audio-2-mini-Think \
  --revision 729b52fd9f4d5075fe60197e0a710be7392680a1 \
  --local-dir "$MODEL_ROOT/Step-Audio-2-mini-Think"

hf download moonshotai/Kimi-Audio-7B-Instruct \
  --revision 9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b \
  --local-dir "$MODEL_ROOT/Kimi-Audio-7B-Instruct"

hf download microsoft/Phi-4-multimodal-instruct \
  --revision 93f923e1a7727d1c4f446756212d9d3e8fcc5d81 \
  --local-dir "$MODEL_ROOT/Phi-4-multimodal-instruct"

hf download XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --revision c359441c22c2a1c74be5f99a91e83392680e9cc8 \
  --local-dir "$MODEL_ROOT/MiMo-Audio-7B-Instruct"

hf download XiaomiMiMo/MiMo-Audio-Tokenizer \
  --local-dir "$MODEL_ROOT/MiMo-Audio-Tokenizer"
```

MiMo tokenizer 在現有 manifest 沒有獨立記錄 commit；第一次下載後必須補記 repo commit SHA 再凍結，不要在正式重現時長期使用浮動 `main`。

### 4.3 Gemma 3n：需先申請存取

先登入 Hugging Face，開啟 `google/gemma-3n-E4B-it` 頁面並接受 Google Gemma license。獲准後：

```bash
hf download google/gemma-3n-E4B-it \
  --revision c1221e9c62e34a43ab7ffacd1be0ea71f126ef10 \
  --local-dir "$MODEL_ROOT/gemma-3n-E4B-it"
```

若出現 401/403，需確認登入帳號已獲 repo access，且 token 有 read permission；不要關閉 SSL 驗證。

## 5. 每類模型使用獨立 Python environment

| Environment | 模型 | 關鍵版本 |
|---|---|---|
| `modern-omni` | Qwen3-Omni、Gemma 4、AF3 | Python 3.11；成功基線為 Torch 2.7.1+cu128、Transformers 5.12.1 |
| `minicpm451` | MiniCPM-o 4.5 | Transformers 4.51.0；Torch 2.3–2.8；`minicpmo-utils[all]>=1.0.5` |
| `step449` | Step Audio 2 | Python >=3.10；Transformers 4.49.0；官方 Step runtime |
| `phi448` | Phi-4 multimodal | Python 3.10；Torch 2.6.0；Transformers 4.48.2；FlashAttention 2.7.4.post1 |
| `kimi` | Kimi Audio | 官方 Docker優先；否則 Torch 2.6.0 + 官方 requirements |
| `mimo` | MiMo Audio | CUDA >=12.0；Torch/Torchaudio 2.6.0；Transformers 4.49.0；Triton 3.2.0 |
| `gemma3n453` | Gemma 3n | Transformers 4.53.0 |

### 5.1 Qwen3-Omni、Gemma 4、AF3

```bash
python3.11 -m venv /data/venvs/modern-omni
source /data/venvs/modern-omni/bin/activate
python -m pip install -U pip

# 依對方 driver/CUDA 選擇正確 PyTorch wheel。
python -m pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1
python -m pip install transformers==5.12.1 accelerate librosa soundfile scipy pillow
python -m pip install qwen-omni-utils

# 選用；GPU、CUDA、Torch 必須相容。
python -m pip install flash-attn --no-build-isolation
```

Qwen：

- counting 只要文字輸出時使用 `return_audio=False`。
- Instruct 可在 load 後呼叫 `model.disable_talker()`，約省 10 GB VRAM。
- 官方 BF16/FlashAttention 表中，15 秒 video 情境 Instruct 約 78.85 GB、Thinking 約 68.74 GB；純短音訊可能較低，但仍從 batch size 1 開始。
- 需 `ffmpeg` 及 `qwen-omni-utils` 正確處理 audio/message payload。

Gemma 4：

- 使用 `AutoProcessor` + `AutoModelForMultimodalLM`。
- audio item 放在文字 instruction 之後。
- model card 標示 audio 最長 30 秒；較長輸入需明確切片。

AF3：

- 授權為 **NVIDIA OneWay Noncommercial License**，商業用途前先審查。
- 本專案以 checkpoint 原生 `float32` 才避開 convolution/LayerNorm dtype 錯誤；不要未測試就強制全模型 BF16。
- 以 30 秒 windows 處理，單 sample 總上限 10 分鐘。
- 歷史上常漏 `FINAL_COUNT:` marker；必須保存 raw output。
- FlashAttention 2 與 `torch.compile` 不要同時啟用；沒有 FlashAttention 時可用 SDPA。

### 5.2 MiniCPM-o 4.5

```bash
python3.11 -m venv /data/venvs/minicpm451
source /data/venvs/minicpm451/bin/activate
python -m pip install -U pip
python -m pip install \
  "transformers==4.51.0" accelerate \
  "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0" \
  "minicpmo-utils[all]>=1.0.5"
```

以 `AutoModel.from_pretrained(..., trust_remote_code=True)` 載入；audio understanding 通常使用 16 kHz mono。只需文字輸出時關掉不必要的 audio generation。

### 5.3 Step-Audio 2 mini / Think

```bash
python3.10 -m venv /data/venvs/step449
source /data/venvs/step449/bin/activate
python -m pip install -U pip
python -m pip install torch torchaudio
python -m pip install \
  transformers==4.49.0 librosa onnxruntime s3tokenizer diffusers hyperpyyaml

git clone https://github.com/stepfun-ai/Step-Audio2.git /data/runtime/Step-Audio2
git -C /data/runtime/Step-Audio2 checkout 76e272b56c3917a8d7188f18bbb5a65dfc8a0845
```

- mini 與 mini-Think 是兩個獨立 checkpoint。
- 大量 serving 可考慮官方 `stepfun2025/vllm:step-audio-2-v20250909` image。
- Think 版本能生成，但 strict marker compliance 很差，目前 formal 排除。

### 5.4 Phi-4 multimodal

```bash
python3.10 -m venv /data/venvs/phi448
source /data/venvs/phi448/bin/activate
python -m pip install -U pip
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  transformers==4.48.2 accelerate==1.3.0 \
  soundfile==0.13.1 pillow==11.1.0 scipy==1.15.2 \
  backoff==2.2.1 peft==0.13.2
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

使用 `AutoProcessor` / `AutoModelForCausalLM` 並設定 `trust_remote_code=True`。Ampere 或更新 GPU 可用 FlashAttention；舊 GPU 改用 eager attention。一般 audio 建議不超過 40 秒。

### 5.5 Kimi Audio

官方建議 Docker：

```bash
docker pull moonshotai/kimi-audio:v0.1
```

若原生安裝：

```bash
git clone --recurse-submodules https://github.com/MoonshotAI/Kimi-Audio /data/runtime/Kimi-Audio
git -C /data/runtime/Kimi-Audio checkout 349251e1d8f4f98d58fda59246381faecd7392e0

python3.11 -m venv /data/venvs/kimi
source /data/venvs/kimi/bin/activate
python -m pip install -U pip
python -m pip install -r /data/runtime/Kimi-Audio/requirements.txt
python -m pip install -e /data/runtime/Kimi-Audio
```

checkpoint 必須保留整個 repo，不只 LLM shards：還包含 Whisper、vocoder、audio detokenizer。官方 requirements pin `torch==2.6.0`、`torchaudio==2.6.0`、`flash_attn==2.7.4.post1`、`deepspeed==0.16.9` 等；若失敗先對照官方 Dockerfile，不要隨意升級整組套件。

### 5.6 MiMo Audio

```bash
git clone https://github.com/XiaomiMiMo/MiMo-Audio.git /data/runtime/MiMo-Audio
git -C /data/runtime/MiMo-Audio checkout 691ce54144a6844cc641fd96046a6ba20776c8b0

python3.11 -m venv /data/venvs/mimo
source /data/venvs/mimo/bin/activate
python -m pip install -U pip
python -m pip install -r /data/runtime/MiMo-Audio/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

MiMo 必須同時提供 Instruct checkpoint、Tokenizer checkpoint 與官方 runtime。requirements 關鍵 pin：`torch==2.6.0`、`torchaudio==2.6.0`、`transformers==4.49.0`、`triton==3.2.0`，CUDA >=12.0。

### 5.7 Gemma 3n

```bash
python3.11 -m venv /data/venvs/gemma3n453
source /data/venvs/gemma3n453/bin/activate
python -m pip install -U pip
python -m pip install torch torchvision torchaudio
python -m pip install transformers==4.53.0 accelerate librosa soundfile pillow
```

使用 `AutoProcessor` + `Gemma3nForConditionalGeneration`。原機器因 gated access 沒有成功下載 checkpoint；對方下載、驗 shard 並完成單題 smoke 後，才能標示可用。

## 6. 下載後完整性與 smoke

```bash
find "$MODEL_ROOT" -xtype l -print
find "$MODEL_ROOT" -name '*.safetensors' -size 0 -print
find "$MODEL_ROOT" -name 'model.safetensors.index.json' -print
du -sh "$MODEL_ROOT"/*
```

預期沒有 broken symlink、0-byte weight，且每個 index 的 `weight_map` shard 都存在。config、processor/tokenizer、generation config 及必要自訂 `.py` code 也要存在。

下載後設定 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，用一個短 WAV 完成：

1. processor 載入；
2. audio decode/resample；
3. model load；
4. deterministic generation；
5. 保存 raw output；
6. 專案 parser 驗證。

## 7. 建議順序

1. Qwen3-Omni Instruct：>=80 GB GPU、batch size 1、`return_audio=False`。
2. MiniCPM-o 4.5：獨立 `minicpm451` environment。
3. Gemma 4 與 AF3；AF3 先採 float32/SDPA 保守設定。
4. 再逐個處理 Step、Phi、Kimi、MiMo，一次只新增一個 environment。
5. 最後處理 gated Gemma 3n。
6. 每個環境保存 `pip freeze`；每個模型保存 repo ID、revision SHA、config SHA-256、權重檔名與大小。

## 8. 這個任務不必額外下載

原機器另有 Qwen2.5-14B-Instruct、Qwen3-32B、Qwen3-8B、Qwen2.5-Omni-7B、Whisper large-v3 與兩個 Qwen3-8B LoRA。它們不是目前 11-model audio counting roster 的必要項目。若沒有純文字 baseline、ASR 或 adapter 實驗，不必跟著下載，可省約 130 GB 以上。

## 9. 最重要的注意事項

- 不要把所有模型套件裝進同一個 environment。
- 不要用浮動 `main` 取代已有的 revision SHA。
- 不要在 CPU/login node 載入大型模型；下載與 GPU inference 分離。
- 不要把 token 寫進程式、設定或 log。
- `trust_remote_code=True` 只對已 pin、已審查的本機 checkpoint 使用。
- sample rate、聲道、長度及 chat template 都是模型介面的一部分，不能共用一套前處理硬套所有模型。
- 先保存 raw output，再 parsing；AF3 與 Step Think 的 marker failure 是已知行為。
- 顯存數字只供短音訊、既有 runtime 參考；長音訊、大 batch、audio output 或 video 都會增加需求。
- 正式批次前，先各做一題 smoke，再做 4–8 題 pilot。
