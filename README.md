# Open-LLaVA-NeXT
A repoduction of **LLaVA-NeXT** series for facilating the large multi-modal model community.

## 💡 Highlights
- 🔥 All data used are **open source**.
- 🔥 Able to reproduce the results of **[LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/)** on all benchmarks.
- 🔥 Primarily based on the **[LLaVA](https://github.com/haotian-liu/LLaVA)** code base, very easy to get started with training.

## 🤖 Model Zoo

See more details in [ModelZoo.md](docs/ModelZoo.md). 

| Name | LLM | Checkpoint | SEED-image | SQA-image | MMBench | MMBench-CN | TextVQA | VizWiz | GQA | VQA-v2 | POPE | MME |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-NeXT-Vicuna-7B | Vicuna-7B | [LLaVA-NeXT-Vicuna-7B](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) | 70.2 | 70.1 | 67.4 | 60.6 | 64.9 | 57.6 | 64.2 | 81.8 | 86.5 | 1519 |
| Open-LLaVA-NeXT-Vicuna-7B | Vicuna-7B | [Open-LLaVA-NeXT-Vicuna-7B]() | 70.9 | 71.2 | 68.0 | 60.7 | 67.3 | 59.4 | 64.2 | 81.7 | 86.3 | 1489 |


## Install

1. Clone this repository and navigate to Open-LLaVA-NeXT folder
```bash
git clone GitHub - xiaoachen98/Open-LLaVA-NeXT: A repoduction of LLaVA-NeXT series for facilating the large mu
cd Open-LLaVA-NeXT
```

2. Install Package
```Shell
conda create -n openllava1.6 python=3.10 -y
conda activate openllava1.6
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data Preparation

You should follow this instruction **[Data.md](docs/Data.md)** to manage the datasets.

## Train

Open-LLaVA-NeXT training consists of two stages: (1) feature alignment stage: use 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage:  finetune the projector, vision encoder and LLM with 1M **completely open source** data. Detailed data composition is in [Visual Instruction Tuning](Build software better, together).

If you just want to fine-tune for Dynamic High Resolution, you can choose to skip the pretrain stage and reuse the pretrained connector of LLaVA-1.5, just as done in the [LLaVA-NeXT Blog](https://llava-vl.github.io/blog/2024-01-30-llava-next/).

Open-LLaVA-NeXT is trained on 16 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Hyperparameters
We use a similar set of hyperparameters as LLaVA in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Projector lr | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-LLaVA-NeXT-7B | 256 | 1e-3 | 1 | 4096 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size |  LLM lr |  Projector lr |  Vision Tower lr | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Open-LLaVA-NeXT-7B | 128 | 2e-5 | 2e-5 | 2e-6 | 1 | 4096 | 0 |


### Pretrain (feature alignment)

Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Pretrain takes around 5 hours for Open-LLaVA-NeXT-7B on 16 x A100 (80G).

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](scripts/v1_6/train/7b/pretrain.sh).

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.

### Visual Instruction Tuning



You may download our pretrained projectors in [Model Zoo](docs/ModelZoo.md). 

Visual instruction tuning takes around 20 hours for Open-LLaVA-NeXT-7B on 16x A100 (80G).

Training script with DeepSpeed ZeRO-2: [`finetune.sh`](scripts/v1_6/train/7b/finetune.sh).

New options to note:

- `--unfreeze_mm_vision_tower True`: finetune vision tower.
- `--mm_vision_tower_lr 2e-6`: vision tower learning rate.
- `--image_aspect_ratio anyres`: Process an image with variable resolutions.
- `--mm_patch_merge_type spatial_unpad`: This unpads a PyTorch tensor of a padded and resized image, and by inserting learnable newline vectors into image tokens, the model becomes aware of two-dimensional spatial information. This is used to process image token.


## Evaluation

See [Evaluation.md](docs/Evaluation.md).

## ❤️ Acknowledgments
- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their brilliant contributions to the community! We just can't wait to use LLaVA-NeXT.
- [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!
