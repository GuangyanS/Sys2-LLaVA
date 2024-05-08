deepspeed --master_port=24999 val_ds.py \
  --version="xinlai/LISA-7B-v1" \
  --dataset_dir='/home/dxleec/gysun/datasets/refer_seg' \
  --vision_pretrained="/home/dxleec/gysun/init_weights/segment_anything/sam_vit_h_4b8939.pth" \
  --dataset="reasonseg" \
  --val_dataset="refcoco+|unc|val" \
  --sample_rates="1" \
  --exp_name="lisa-7b" \
  --eval_only