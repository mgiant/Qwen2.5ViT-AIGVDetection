# download and unzip
# https://huggingface.co/datasets/mgiant/magic_videos

python preprocess/video2frame.py -v data/videos/magic/fake/hailuo-T2V-01 -i data/magic/fake_hailuo
python preprocess/video2frame.py -v data/videos/magic/fake/jimeng-S2.0 -i data/magic/fake_jimeng2.0
python preprocess/video2frame.py -v data/videos/magic/fake/jimeng-S3.0 -i data/magic/fake_jimeng3.0
python preprocess/video2frame.py -v data/videos/magic/fake/pexels_wan2.1-T2V-1.3B -i data/magic/fake_wan1.3B_pexels
python preprocess/video2frame.py -v data/videos/magic/fake/step-video -i data/magic/fake_stepvideo
python preprocess/video2frame.py -v data/videos/magic/fake/wanx2.1 -i data/magic/fake_wan2.1
python preprocess/video2frame.py -v data/videos/magic/real/real_videos_pexels_720p -i data/magic/real_pexels
python preprocess/video2frame.py -v data/videos/magic/real/real_videos_mixkit_720p -i data/magic/real_mixkit