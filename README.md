# Open-LLaVA-NeXT

An open-source implementation of **LLaVA-NeXT** series for facilitating the large multi-modal model community.

**Resources:** [[🤗HuggingFace](https://huggingface.co/collections/Lin-Chen/open-llava-next-665051533fa1a30553fcee8d)]

## 💡 Highlights

- 🔥 All training data and checkpoints at each stage are open-sourced, friendly for research usage.
- 🔥 Able to reproduce the results of **[LLaVA-NeXT](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)**.
- 🔥 Based on the **[LLaVA](https://github.com/haotian-liu/LLaVA)** codebase with minimal modification, easy to follow.

## 🤖 Model Zoo

See more details in [ModelZoo.md](docs/ModelZoo.md).

| Name | ViT | LLM | Weights | MME | SEED | SQA | MMB | MMB-CN | TextVQA | GQA |
|---|---|---|---|---|---|---|---|---|---|---|
| llava-next-vicuna-7b | CLIP-L-336 | Vicuna-7B | [SFT](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) | 1519 | 70.2 | 70.1 | 67.4 | 60.6 | 64.9 | 64.2 |
| open-llava-next-vicuna-7b| CLIP-L-336 | Vicuna-7B | [PT](https://huggingface.co/Lin-Chen/open-llava-next-vicuna-7b/tree/main/pretrain), [SFT](https://huggingface.co/Lin-Chen/open-llava-next-vicuna-7b) | 1540 | 71.1 | 70.7 | 68.5 | 60.7 | 67.2 | 64.3 |
| llava-next-llama3-8b| CLIP-L-336 | LLaMA3-8B | [SFT](https://huggingface.co/lmms-lab/llama3-llava-next-8b) | 1591 | 72.7 | 73.4 | 72.6 | 69.0 | 65.0 | 65.5 |
| open-llava-next-llama3-8b| CLIP-L-336 | LLaMA3-8B | [PT](https://huggingface.co/Lin-Chen/open-llava-next-llama3-8b), [SFT](https://huggingface.co/Lin-Chen/open-llava-next-llama3-8b) | 1552 | 74.4 | 77.3 | 74.4 | 70.4 | 69.8 | 65.9 |


## 👨‍💻 ToDo

- [x] Reproduce LLaVA-Next-LLaMA3-8B
- [ ] Reproduce LLaVA-Next-Nous-Yi-34B
- [ ] Support SigLIP and more LLMs with various scales (Need the help from community!)
- [ ] Integrate [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for convenient evaluation

## 🔧 Install

1. Clone this repository and navigate to Open-LLaVA-NeXT folder
```bash
git clone https://github.com/xiaoachen98/Open-LLaVA-NeXT.git
cd Open-LLaVA-NeXT
```

2. Install Package
```Shell
conda create -n llava-next python=3.10 -y
conda activate llava-next
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

1. Install additional packages for training
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data Preparation

You should follow this instruction **[Data.md](docs/Data.md)** to manage the training datasets.

## Training Overview

Open-LLaVA-NeXT training consists of two stages: (1) feature alignment stage: use 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage:  finetune the entire model with 1M **completely open source** data. Detailed data statics is provided in [Visual Instruction Tuning](https://github.com/xiaoachen98/Open-LLaVA-NeXT?tab=readme-ov-file#visual-instruction-tuning). We take the Vicuna-v1.5-7B variant as example to present the training  and evaluation details.

Open-LLaVA-NeXT series are trained on A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. And utilizing DeepSpeed ZeRO-3 can further reduce the memory requirements. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Hyperparameters
We use a same set of hyperparameters as LLaVA in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Projector lr | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-LLaVA-NeXT-7B | 256 | 1e-3 | 1 | 4096 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size |  LLM lr |  Projector lr |  Vision Tower lr | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Open-LLaVA-NeXT-7B | 128 | 2e-5 | 2e-5 | 2e-6 | 1 | 4096 | 0 |

### Pretrain

Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Pretrain takes around 5 hours for Open-LLaVA-NeXT-7B on 16 x A100 (80G).

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](scripts/v1_6/train/7b/pretrain.sh).

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.

### Visual Instruction Tuning

1. Prepare data
You should follow the instructions for data preparation in [Data](docs/Data.md).
2. Prepare MLP projectors
You may download our pretrained projectors in [Model Zoo](docs/ModelZoo.md), or specify your own MLP projector after pre-training.
3. Start training
Visual instruction tuning takes around 20 hours for Open-LLaVA-NeXT-7B on 16x A100 (80G).

Training script with DeepSpeed ZeRO-2: [`finetune.sh`](scripts/v1_6/train/7b/finetune.sh).

New options to note:

- `--unfreeze_mm_vision_tower True`: finetune vision tower.
- `--mm_vision_tower_lr 2e-6`: learning rate of vision tower.
- `--image_aspect_ratio anyres`: Process an image with variable resolutions.
- `--mm_patch_merge_type spatial_unpad`: This unpads a PyTorch tensor of a padded and resized image, and by inserting learnable newline vectors into image tokens, the model becomes aware of two-dimensional spatial information. This is used to process image token.

## Evaluation

See [Evaluation.md](docs/Evaluation.md).

## ❤️ Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their brilliant contributions to the community! We just can't wait to use LLaVA-NeXT.
- [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V): Thanks for their code about finetuning the vision tower.
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): the amazing open-sourced suit for evaluating various LMMs!
