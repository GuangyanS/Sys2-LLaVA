cd /ibex/ai/home/wangc0g/gysun/datasets/eval/vizwiz

mkdir test
cd test

wget https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip
unzip Annotations.zip
rm Annotations.zip

wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip
unzip test.zip
rm test.zip