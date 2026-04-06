# download and unzip
# https://d14whct5a0wtwm.cloudfront.net/moviegen/MovieGenVideoBench.tar.gz
python preprocess/video2frame.py -v data/videos/moviegen/water_mark_out -i data/moviegen-2k/fake_moviegen

# download and unzip
# https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/blob/main/zip_folder/panda/panda_videos_21.zip
mkdir -p data/videos/panda_videos_1k
find data/videos/panda_videos -maxdepth 1 -type f | shuf -n 1000 | xargs -I {} mv "{}" data/videos/panda_videos_1k/
python preprocess/video2frame.py -v data/videos/panda_videos_1k -i data/moviegen-2k/real_panda
