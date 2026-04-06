# real_kinetics_70k
# download and unzip 
# (from part_0 to part_69) https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt
# example: https://s3.amazonaws.com/kinetics/400/train/part_0.tar.gz
python preprocess/video2frame.py -v data/videos/kinetics -i data/real_images/real_kinetics

# fake_apple_t2v, fake_causvid, fake_gen3, fake_hunyuan, fake_jimeng, fake_kling, fake_luma, fake_mira, fake_opensorav1-1, fake_pika, fake_repvideo, fake_sora, fake_wanx21, fake_animatediff-v2, fake_videocrafter2, real_msvd_images
# download and unzip
# apple t2v: https://drive.google.com/file/d/19ENw3mbyhz-JjW4ddWypJ8x6xvWOHsyr
# causvid https://drive.google.com/drive/folders/1SHD4CFuzBXLrsk1_fmQFBz7wvUvedY2O
# gen3 https://drive.google.com/drive/folders/1AFV48EOAXydz2ZB-q2ml7b0ojYrAsl-K
# hunyuan https://drive.google.com/file/d/1gjg9fO6_k5OIRAZtqIIxXd6-k4c1Tjb4
# jimeng https://drive.google.com/drive/folders/1lPJp8qspt5h6H6OTX37-BSl73DlLezOL
# kling https://drive.google.com/drive/folders/1g5Y9j2gb9I5FUg4Ql28jSCCPH-Pv-HDg
# luma https://drive.google.com/file/d/1NuL9oRIMPuPk98PfI0lA34LL6XLtjMnr
# mira https://drive.google.com/file/d/1lx0evF0HN0jY3FQ41RhQL9UJbOa-gve6
# opensora-v1.1 https://drive.google.com/file/d/1mGxjDIf7IT_mNibG8Nmg3E1WcXYRVDoo
# pika https://drive.google.com/file/d/1G2VVD5ArLxYtKeAVdANnxNNAPlP2bbZO
# repvideo https://drive.google.com/file/d/1H4eL4SOgidlOZeFPhX_4nzef7OGqpOQM
# sora https://drive.google.com/drive/folders/1ZxANm9HssrOY7aPAKRi579eJaQV6fsbk
# wan21 https://drive.google.com/drive/folders/1GHH7xOQCPb0kRjlyzH9APaddAw_hFRla
# animatediff-v2 https://drive.google.com/file/d/1a9dPyArEWt61NS3E2VDws8wMAXI-MX04
# videocrafter2 https://drive.google.com/file/d/17podJKS0tbfUS8dVAPNyDv4vYo4dIDqL

# apple t2v 4715 videos
tar -xzvf "data/zipfiles/STIV (Apple).tar.gz" -C data/videos/vbench/
mv data/videos/vbench/apple_t2v/*/*.mp4 data/videos/vbench/apple_t2v/
rm -r data/videos/vbench/apple_t2v/*/
python preprocess/video2frame.py -v data/videos/vbench/apple_t2v -i data/fake_images/fake_apple_t2v

# causvid 4720 videos
tar -xzvf data/zipfiles/causvid_24fps.tar.gz -C data/videos/vbench/
python preprocess/video2frame.py -v data/videos/vbench/causvid_24fps -i data/fake_images/fake_causvid

# gen3 4706 videos
tar -xzvf data/zipfiles/gen3_all_dimension.tar.gz -C data/videos/vbench/
mv data/videos/vbench/all_dimension data/videos/vbench/gen3
python preprocess/video2frame.py -v data/videos/vbench/gen3 -i data/fake_images/fake_gen3

# hunyuan 4715 videos
mv data/zipfiles/hunyuan_all_dimension data/videos/vbench
python preprocess/video2frame.py -v data/videos/vbench/hunyuan_all_dimension -i data/fake_images/fake_hunyuan

# jimeng 6214
unzip data/zipfiles/Jimeng_all_dimension.zip -d data/videos/vbench/
mv data/videos/vbench/all_dimension data/videos/vbench/jimeng
python preprocess/video2frame.py -v data/videos/vbench/jimeng -i data/fake_images/fake_jimeng

# kling 4679
tar -xzvf data/zipfiles/Kling_all.tar.gz -C data/videos/vbench/
python preprocess/video2frame.py -v data/videos/vbench/Kling_filtered -i data/fake_images/fake_kling

# luma 4680
tar -xvf data/zipfiles/Luma_all_dimension.tar -C data/videos/vbench/
mv data/videos/vbench/all_dimension data/videos/vbench/luma
python preprocess/video2frame.py -v data/videos/vbench/luma -i data/fake_images/fake_luma

# mira 4720
tar -xvf data/zipfiles/Mira-384.tar -C data/videos/vbench/
mv data/videos/vbench/Mira-384/*/*.mp4 data/videos/vbench/Mira-384/
rm -r data/videos/vbench/Mira-384/*/
python preprocess/video2frame.py -v data/videos/vbench/Mira-384 -i data/fake_images/fake_mira

# opensorav1-1 4720
tar -xvf data/zipfiles/OpenSorav1-1.tar -C data/videos/vbench/
mv data/videos/vbench/OpenSorav1-1/*/*.mp4 data/videos/vbench/OpenSorav1-1/
rm -r data/videos/vbench/OpenSorav1-1/*/
python preprocess/video2frame.py -v data/videos/vbench/OpenSorav1-1 -i data/fake_images/fake_opensorav1-1

# pika 4715
tar -xzvf data/zipfiles/pika_1_all_dimension.tar.gz -C data/videos/vbench/
mv data/videos/vbench/all_dimension data/videos/vbench/pika
python preprocess/video2frame.py -v data/videos/vbench/pika -i data/fake_images/fake_pika

# repvideo 4720
unzip data/zipfiles/RepVideo.zip -d data/videos/vbench/
mv data/videos/vbench/RepVideo data/videos/vbench/repvideo
python preprocess/video2frame.py -v data/videos/vbench/repvideo -i data/fake_images/fake_repvideo

# sora 4720
tar -xzvf data/zipfiles/sora.tar.gz -C data/videos/vbench/
python preprocess/video2frame.py -v data/videos/vbench/sora -i data/fake_images/fake_sora

# wanx21 4725
tar -xzvf data/zipfiles/wanx21.tar.gz -C data/videos/vbench/
mv data/videos/vbench/wanx21/*/*.mp4 data/videos/vbench/wanx21/
rm -r data/videos/vbench/wanx21/*/
python preprocess/video2frame.py -v data/videos/vbench/wanx21 -i data/fake_images/fake_wanx21

# animatediff-v2 4715
unrar x data/zipfiles/AnimateDiff-v2.rar -d data/videos/vbench
mv data/videos/vbench/AnimateDiff-v2/*/*.mp4 data/videos/vbench/AnimateDiff-v2/
rm -r data/videos/vbench/AnimateDiff-v2/*/
python preprocess/video2frame.py -v data/videos/vbench/AnimateDiff-v2 -i data/fake_images/fake_animatediff-v2

# videocrafter2 4720
mkdir data/videos/vbench/videocrafter2
tar -xvf data/zipfiles/videocrafter-2.tar -C data/videos/vbench/videocrafter2
mv data/videos/vbench/videocrafter2/*/*.mp4 data/videos/vbench/videocrafter2/
rm -r data/videos/vbench/videocrafter2/*/
python preprocess/video2frame.py -v data/videos/vbench/videocrafter2 -i data/fake_images/fake_videocrafter2


# python preprocess/video2frame.py -v data/videos/vbench/apple_t2v -i data/fake_images/fake_apple_t2v
# python preprocess/video2frame.py -v data/videos/vbench/causvid_24fps -i data/fake_images/fake_causvid
# python preprocess/video2frame.py -v data/videos/vbench/gen3 -i data/fake_images/fake_gen3
# python preprocess/video2frame.py -v data/videos/vbench/hunyuan_all_dimension -i data/fake_images/fake_hunyuan
# python preprocess/video2frame.py -v data/videos/vbench/jimeng -i data/fake_images/fake_jimeng
# python preprocess/video2frame.py -v data/videos/vbench/Kling_filtered -i data/fake_images/fake_kling
# python preprocess/video2frame.py -v data/videos/vbench/luma -i data/fake_images/fake_luma
# python preprocess/video2frame.py -v data/videos/vbench/Mira-384 -i data/fake_images/fake_mira
# python preprocess/video2frame.py -v data/videos/vbench/OpenSorav1-1 -i data/fake_images/fake_opensorav1-1
# python preprocess/video2frame.py -v data/videos/vbench/pika -i data/fake_images/fake_pika
# python preprocess/video2frame.py -v data/videos/vbench/repvideo -i data/fake_images/fake_repvideo
# python preprocess/video2frame.py -v data/videos/vbench/sora -i data/fake_images/fake_sora
# python preprocess/video2frame.py -v data/videos/vbench/wanx21 -i data/fake_images/fake_wanx21
# python preprocess/video2frame.py -v data/videos/vbench/AnimateDiff-v2 -i data/fake_images/fake_animatediff-v2
# python preprocess/video2frame.py -v data/videos/vbench/videocrafter2 -i data/fake_images/fake_videocrafter2