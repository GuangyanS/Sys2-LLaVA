# Visual Agents as Fast and Slow Thinkers

## Requirements
Create a new environment and install the required dependencies:
```
conda create -n sys2 python=3.10 -y
conda activate sys2
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
export PYTHONPATH=$PYTHONPATH:path/to/repo
```

## Training

### Switch Adapter

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

For the pre-training stage, enter the SwiLLaVA script folder and run
```
sh pretrain.sh
```

For the instruction tuning stage, enter the SwiLLaVA script folder and run
```
sh finetune_lora.sh
```

### ROI Adapter

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

Start Training: The finetuning process takes around 20 hours on 8*A100 (80G) for LLaVA-1.5-13B. We finetune LLaVA-1.5 using Deepspeed Zero-3, you can directly run the scripts to launch training:

```
bash ./scripts/v1_5/finetune_lora.sh
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
To train the SegLLaVA, use the command:
```
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LLaVA-v1.5" \
  --dataset_dir='./dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="sem_seg||refer_seg" \
  --sample_rates="3,1" \
  --exp_name="seg-llava-7b"
```

## Evaluation

We use the same setting of LLaVA v1.5. We evaluate models on a diverse set of 8 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) for more details. The scripts are under folder `eval`.

```
@misc{sun2024visualagentsfastslow,
      title={Visual Agents as Fast and Slow Thinkers}, 
      author={Guangyan Sun and Mingyu Jin and Zhenting Wang and Cheng-Long Wang and Siqi Ma and Qifan Wang and Ying Nian Wu and Yongfeng Zhang and Dongfang Liu},
      year={2024},
      eprint={2408.08862},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.08862}, 
}
```
