import os
import argparse
import pandas as pd


VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm')


def find_all_videos(folder_path):
    """Recursively search for all video files under folder_path."""
    video_files = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if f.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, f))
    return video_files


def infer_label(file_path, applied_label):
    """
    Determine the label for a video file:
      - If applied_label >= 0, use it directly.
      - If applied_label == -1, infer from the file path
        based on whether it contains 'fake' or 'real'.
    """
    if applied_label >= 0:
        return applied_label

    path_lower = file_path.lower()
    if 'fake' in path_lower:
        return 1
    elif 'real' in path_lower:
        return 0
    else:
        return None  # Unable to infer


def generate_csv(folder_path, applied_label, out_root):
    os.makedirs(out_root, exist_ok=True)

    video_files = find_all_videos(folder_path)
    if not video_files:
        print(f"[WARNING] No video files found under {folder_path}")
        return

    content_paths = []
    labels = []
    type_ids = []

    for video_path in video_files:
        label = infer_label(video_path, applied_label)
        if label is None:
            print(f"[SKIP] Cannot infer label, skipping: {video_path}")
            continue

        content_paths.append(video_path)
        labels.append(label)
        type_ids.append('fake' if label else 'real')

    if not content_paths:
        print("[WARNING] No valid video records to write")
        return

    df = pd.DataFrame({
        'content_path': content_paths,
        # 'image_path': content_paths,
        'type_id': type_ids,
        'label': labels,
    })

    out_path = os.path.join(out_root, 'all.csv')
    df.to_csv(out_path, encoding='utf-8', index=False)
    print(f"[DONE] {len(df)} records saved to {out_path}")


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Recursively scan a directory for video files and generate a CSV"
    )
    parser.add_argument('--folder', dest='folder_path', required=True,
                        help='Root directory to scan')
    parser.add_argument('--label', dest='label', type=int, default=-1,
                        help='Forced label (0=real, 1=fake); -1 to auto-infer from path')
    parser.add_argument('--out', dest='out_root', required=True,
                        help='Output directory for CSV files')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    generate_csv(args.folder_path, applied_label=args.label, out_root=args.out_root)