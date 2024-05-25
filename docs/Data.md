## Data

| Data file name | Size |
| --- | ---: |
| [open-llava-next_instruct_mix1M.json](https://huggingface.co/datasets/Lin-Chen/Open-LLaVA-NeXT-mix1M/blob/main/open-llava-next_instruct_mix1M.json) | 1.64 GB |
| [vqa_collection.zip](https://huggingface.co/datasets/Lin-Chen/Open-LLaVA-NeXT-mix1M/blob/main/vqa_collection.zip) | 30.20 GB |

We have made every effort to align our training data with that of LLaVA-NeXT. However, we were unable to access the tens of thousands of real user interaction data that LLaVA-NeXT collected. As a result, we used 200K [ALLaVA-Instruct-VFLAN-4V](https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k/blob/main/image_191-task_1k.zip) data as a substitute. Additionally, since [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) has been included in the training data of most existing LMMs, we chose to retain it to enable fair comparisons with other LMMs.

### Dataset
The dataset, based on [sharegpt4v_mix665k](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json), has been expanded to include ALLaVA-Instruct-VFLAN-4V, DocVQA, SynDog-EN, ChartQA, DVQA, AI2D, and GeoQA+, totaling 1M image-text pairs.

### Prepare Images

First, download all images we used.

- LAION-CC-SBU-558K: [images.zip](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- WebData: [images](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing). Only for academic usage.
- SAM: [images](https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing). We save all files as `.jpg`
- TextVQA: [trainvalimages](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- A [collection](https://huggingface.co/datasets/Lin-Chen/Open-LLaVA-NeXT-mix1M/blob/main/vqa_collection.zip) of several VQA datasets: DocVQA, SynDog-EN, ChartQA, DVQA, AI2D, and GeoQA+.
- ALLaVA-Instruct-VFLAN-4V: [image_191-task_1k](https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k/blob/main/image_191-task_1k.zip)

Then, organize the data as follows:

```none
Open-LLaVA-NeXT
├── ...
├── data
│   ├── llava
│   │   ├── llava_pretrain
│   │   │   ├── images
│   ├── coco
│   │   ├── train2017
│   ├── sam
│   │   ├── images
│   ├── gqa
│   │   ├── images
│   ├── ocr_vqa
│   │   ├── images
│   ├── textvqa
│   │   ├── train_images
│   ├── vg
│   │   ├── VG_100K
│   │   ├── VG_100K_2
│   ├── open-llava-next
│   │   ├── open-llava-next_instruct_mix1M.json
│   ├── web-celebrity
│   │   ├── images
│   ├── web-landmark
│   │   ├── images
│   ├── wikiart
│   │   ├── images
│   ├── allava_vflan
│   │   ├── images
│   │   │   ├── images_191task_1k
│   ├── share_textvqa
│   │   ├── images
│   ├── ai2d
│   │   ├── images
│   ├── chatqa
│   │   ├── train
│   │   │   ├── png
│   ├── docvqa
│   │   ├── train
│   │   │   ├── documents
│   ├── dvqa
│   │   ├── images
│   ├── geoqa+ 
│   │   ├── images
│   ├── synthdog-en
│   │   ├── images
├── ...
```
