# download and unzip
# https://modelscope.cn/datasets/cccnju/Gen-Video/resolve/master/GenVideo-Val.zip

python preprocess/video2frame.py -v data/videos/GenVideo-Val/Real -i data/genvideo_val/real_msrvtt
python preprocess/video2frame.py -v data/videos/GenVideo-Val/Fake/Crafter -i data/genvideo_val/fake_crafter
python preprocess/video2frame.py -v data/videos/GenVideo-Val/Fake/Gen2 -i data/genvideo_val/fake_gen2
python preprocess/video2frame.py -v data/videos/GenVideo-Val/Fake/HotShot -i data/genvideo_val/fake_hotshot
python preprocess/video2frame.py -v data/videos/GenVideo-Val/Fake/Lavie -i data/genvideo_val/fake_lavie
python preprocess/video2frame.py -v data/videos/GenVideo-Val/Fake/ModelScope -i data/genvideo_val/fake_modelscope
python preprocess/video2frame.py -v data/videos/GenVideo-Val/Fake/MoonValley -i data/genvideo_val/fake_moonvalley
python preprocess/video2frame.py -v data/videos/GenVideo-Val/Fake/MorphStudio -i data/genvideo_val/fake_morphstudio
python preprocess/video2frame.py -v data/videos/GenVideo-Val/Fake/Show_1 -i data/genvideo_val/fake_show1
python preprocess/video2frame.py -v data/videos/GenVideo-Val/Fake/Sora -i data/genvideo_val/fake_sora
python preprocess/video2frame.py -v data/videos/GenVideo-Val/Fake/WildScrape -i data/genvideo_val/fake_wildscrape