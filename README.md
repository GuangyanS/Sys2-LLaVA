# Sys2-LLaVA

hello
## Download Eval Dataset

https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md

```
sh download_eval.sh
```

## Environment Setup
Create a new environment and install the required dependencies:
```
python -m venv sys2
source sys2/bin/activate
pip install -r requirements.txt
```

## Switch Adapter
The alignment stage of the VQA LLM uses the 558K subset of the LAION-CC-SBU dataset used by LLaVA which can be downloaded here.

The instruction tuning stage requires several instruction tuning subsets which can be found here.

The instruction tuning data requires images from COCO-2014, COCO-2017, and GQA. After downloading them, organize the data following the structure below:

```
├── coco2014
│   └── train2014
├── coco2017
│   └── train2017
└── gqa
     └── images
```

## ROI Adapter

Initial Weights: We use LLaVA-1.5-7B and LLaVA-1.5-13B for finetuning. You may download these models and put them in the ./checkpoint folder.

Download Data: The dataset structure is the same as used in LLaVA, and we provide json files to modify the original LLaVA training dataset into our dataset. To correctly download the data, please check the instructions.

After downloading all of them, organize the data as follows in ./playground/data:

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

Training Data Preparations: We migrate the brilliant work of LRP++ to detect the correct ROI corresponding to a single question or instruction. You can directly download our generated dataset to reproduce our results from Google Drive. You may also follow the Notebook to prepare your own data.

Evaluations on Various Benchmarks: We follow the Evaluation Docs in LLaVA to conduct our experiments. If you find it laborious and complex, please check LMMs-Eval for faster evaluation.

Start Training: The finetuning process takes around 20 hours on 8*A100 (80G) for LLaVA-1.5-13B. We finetune LLaVA-1.5 using Deepspeed Zero-3, you can directly run the scripts to launch training:

```
bash ./scripts/v1_5/finetune_CoS_13b.sh
```

## Seg Adapter
### Training Data Preparation
The training data consists of 4 types of data:

- Semantic segmentation datasets: ADE20K, COCO-Stuff, Mapillary, PACO-LVIS, PASCAL-Part, COCO Images
  - Note: For COCO-Stuff, we use the annotation file stuffthingmaps_trainval2017.zip. We only use the PACO-LVIS part in PACO. COCO Images should be put into the dataset/coco/ directory.
- Referring segmentation datasets: refCOCO, refCOCO+, refCOCOg, refCLEF (saiapr_tc-12)
  - Note: the original links of refCOCO series data are down, and we update them with new ones.
- Visual Question Answering dataset: LLaVA-Instruct-150k
- Reasoning segmentation dataset: ReasonSeg
```
├── dataset
│   ├── ade20k
│   │   ├── annotations
│   │   └── images
│   ├── coco
│   │   └── train2017
│   │       ├── 000000000009.jpg
│   │       └── ...
│   ├── cocostuff
│   │   └── train2017
│   │       ├── 000000000009.png
│   │       └── ...
│   ├── llava_dataset
│   │   └── llava_instruct_150k.json
│   ├── mapillary
│   │   ├── config_v2.0.json
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   ├── reason_seg
│   │   └── ReasonSeg
│   │       ├── train
│   │       ├── val
│   │       └── explanatory
│   ├── refer_seg
│   │   ├── images
│   │   |   ├── saiapr_tc-12 
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
│   └── vlpart
│       ├── paco
│       │   └── annotations
│       └── pascal_part
│           ├── train.json
│           └── VOCdevkit
```
