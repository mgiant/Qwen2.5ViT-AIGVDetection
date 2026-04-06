# download and unzip
# https://github.com/SparkleXFantasy/MM-Det 

# unzip data/zipfiles/DVF/opensora/0_real.zip -d data/videos/dvf-test/real_internvid
# mv data/videos/dvf-test/real_internvid/0_real/* data/videos/dvf-test/real_internvid/; rm -d data/videos/dvf-test/real_internvid/0_real

# unzip data/zipfiles/DVF/opensora/1_fake.zip -d data/videos/dvf-test/fake_opensora
# mv data/videos/dvf-test/fake_opensora/1_fake/* data/videos/dvf-test/fake_opensora/; rm -d data/videos/dvf-test/fake_opensora/1_fake

# unzip data/zipfiles/DVF/pika/1_fake.zip -d data/videos/dvf-test/fake_pika
# mv data/videos/dvf-test/fake_pika/1_fake/* data/videos/dvf-test/fake_pika/; rm -d data/videos/dvf-test/fake_pika/1_fake

# unzip data/zipfiles/DVF/sora/1_fake.zip -d data/videos/dvf-test/fake_sora
# mv data/videos/dvf-test/fake_sora/1_fake/* data/videos/dvf-test/fake_sora/; rm -d data/videos/dvf-test/fake_sora/1_fake

# unzip data/zipfiles/DVF/stablediffusion/1_fake.zip -d data/videos/dvf-test/fake_stablediffusion
# mv data/videos/dvf-test/fake_stablediffusion/1_fake/* data/videos/dvf-test/fake_stablediffusion/; rm -d data/videos/dvf-test/fake_stablediffusion/1_fake

# unzip data/zipfiles/DVF/stablevideo/1_fake.zip -d data/videos/dvf-test/fake_stablevideo
# mv data/videos/dvf-test/fake_stablevideo/1_fake/* data/videos/dvf-test/fake_stablevideo/; rm -d data/videos/dvf-test/fake_stablevideo/1_fake

# unzip data/zipfiles/DVF/videocrafter1/1_fake.zip -d data/videos/dvf-test/fake_videocrafter1
# mv data/videos/dvf-test/fake_videocrafter1/1_fake/* data/videos/dvf-test/fake_videocrafter1/; rm -d data/videos/dvf-test/fake_videocrafter1/1_fake

# unzip data/zipfiles/DVF/zeroscope/1_fake.zip -d data/videos/dvf-test/fake_zeroscope
# mv data/videos/dvf-test/fake_zeroscope/1_fake/* data/videos/dvf-test/fake_zeroscope/; rm -d data/videos/dvf-test/fake_zeroscope/1_fake

# 7z x data/zipfiles/DVF/zeroscope/0_real.zip -odata/videos/dvf-test/fake_zeroscope
# mv data/videos/dvf-test/fake_zeroscope/0_real/*.mp4 data/videos/dvf-test/real_internvid; rm -d data/videos/dvf-test/fake_zeroscope/0_real

# unzip data/zipfiles/DVF/videocrafter1/0_real.zip -d data/videos/dvf-test/real_internvid
# mv data/videos/dvf-test/real_internvid/0_real/*.mp4 data/videos/dvf-test/real_internvid

python preprocess/video2frame.py -v data/videos/dvf-test/real_internvid -i data/dvf_test_images/real_internvid
python preprocess/video2frame.py -v data/videos/dvf-test/fake_opensora -i data/dvf_test_images/fake_opensora
python preprocess/video2frame.py -v data/videos/dvf-test/fake_pika -i data/dvf_test_images/fake_pika
python preprocess/video2frame.py -v data/videos/dvf-test/fake_sora -i data/dvf_test_images/fake_sora
python preprocess/video2frame.py -v data/videos/dvf-test/fake_stablediffusion -i data/dvf_test_images/fake_stablediffusion
python preprocess/video2frame.py -v data/videos/dvf-test/fake_stablevideo -i data/dvf_test_images/fake_stablevideo
python preprocess/video2frame.py -v data/videos/dvf-test/fake_videocrafter1 -i data/dvf_test_images/fake_videocrafter1
python preprocess/video2frame.py -v data/videos/dvf-test/fake_zeroscope -i data/dvf_test_images/fake_zeroscope