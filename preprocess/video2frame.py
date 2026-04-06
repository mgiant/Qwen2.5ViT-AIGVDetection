import argparse
import subprocess
from pathlib import Path

from tqdm.contrib.concurrent import process_map

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
    ".mts",
    ".m2ts",
    ".3gp",
}

def process_video(video_path):
    video_path = Path(video_path)
    image_path = Path(image_root) / video_path.stem

    if image_path.exists() and any(image_path.iterdir()):
        return

    try:
        image_path.mkdir(parents=True, exist_ok=True)
        escaped_image_path = str(image_path).replace("%", "%%") # some video names contain '%'
        output_pattern = f"{escaped_image_path}/%d.jpg"
        cmd_list = [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-i",
            str(video_path),
            "-vf",
            "fps=2",
            output_pattern,
        ]
        subprocess.run(cmd_list, check=True, capture_output=False, text=False)
    except Exception as e:
        print(f"decoding error on {video_path}: {e}")
            

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--video_root' , dest='video_root')
    parser.add_argument('-i', '--image_root', dest='image_root')
    return parser.parse_args()

if __name__ == '__main__':
    print("Getting frames!!")
    args = get_arguments()
    video_root = Path(args.video_root)
    image_root = args.image_root

    all_videos = sorted(
        str(path)
        for path in video_root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    print(f'{len(all_videos)} videos in {args.video_root}')

    process_map(process_video, all_videos, max_workers=8)
