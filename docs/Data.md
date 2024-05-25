## Data

| Data file name | Size |
| --- | ---: |
| [llava-next_mix1M.json]() | 1.53 GB |

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
- A [collection]() of the datasets: DocVQA, SynDog-EN, ChartQA, DVQA, AI2D, and GeoQA+.
- ALLaVA-Instruct-VFLAN-4V: [image_191-task_1k](https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k/blob/main/image_191-task_1k.zip)

Then, organize the data as follows in `projects/ShareGPT4V/data`:

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
│   ├── Open-LLaVA-NeXT
│   │   ├── llava-next_mix1M.json
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
