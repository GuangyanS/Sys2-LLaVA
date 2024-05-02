CUDA_VISIBLE_DEVICES="0,1,2,3" python merge_lora_weights_and_save_hf_model.py \
  --version="liuhaotian/llava-v1.5-7b" \
  --weight="/home/dxleec/gysun/init_weights/LISA-Results/lisa-7b/pytorch_model.bin" \
  --save_path="LISA-Results/lisa-7b/LISA-7B-v1-5"