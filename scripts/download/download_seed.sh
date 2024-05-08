cd /ibex/ai/home/wangc0g/gysun/eval/seed_bench

git clone https://huggingface.co/datasets/AILab-CVC/SEED-Bench

mv SEED-Bench/SEED-Bench-image.zip .
unzip SEED-Bench-image.zip
rm SEED-Bench-image.zip

cd SEED-Bench
unzip v1_video.zip.001
mv v1_video ../