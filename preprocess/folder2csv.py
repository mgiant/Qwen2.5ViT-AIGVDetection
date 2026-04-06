import os
import argparse

import pandas as pd
from pandas import Series


def count_images_in_folder(folder_path):
    image_names = []
    image_files = []
    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_files.append(file_name)
        file_stem = os.path.splitext(file_name)[0]
        if file_stem.isdigit():
            image_names.append(int(file_stem))

    if image_names:
        image_names.sort()
    else:
        image_names = list(range(1, len(image_files) + 1))

    return len(image_files), image_names, image_files


def generate_csv(folder_path, applied_label):
    all_labels = []
    all_save_path = []
    all_frame_counts = []
    all_frame_seq_counts = []
    all_content_paths = []
    all_type_labels = []

    for source in sorted(os.listdir(folder_path)):
        source_path = os.path.join(folder_path, source)
        if not os.path.isdir(source_path):
            continue
        else:
            print(f'source {source}')

        labels = []
        save_path = []
        frame_counts = []
        frame_seq_counts = []
        content_paths = []
        type_labels = []

        if applied_label == -1:
            if 'real' not in source.lower() and 'fake' not in source.lower():
                print(f'skip {source}, cannot infer label')
                continue
            label = 1 if 'fake' in source.lower() else 0
        else:
            label = applied_label
        type_label = 'fake' if label else 'real'

        for subfolder_name in sorted(os.listdir(source_path)):
            subfolder_path = os.path.join(source_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                temp_frame_count, temp_frame_seqs, image_files = count_images_in_folder(subfolder_path)

                if temp_frame_count == 0:
                    print(source, subfolder_name, 'has no frames')
                    continue

                frame_path = os.path.join(subfolder_path, image_files[0])

                labels.append(label)
                type_labels.append(type_label)
                frame_counts.append(temp_frame_count)
                frame_seq_counts.append(temp_frame_seqs)
                save_path.append(frame_path)
                content_paths.append(subfolder_path)
            elif os.path.isfile(subfolder_path) and subfolder_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                labels.append(label)
                type_labels.append(type_label)
                frame_counts.append(1)
                frame_seq_counts.append([1])
                save_path.append(subfolder_path)
                content_paths.append(subfolder_path)

        if len(content_paths) == 0:
            print(source, 'has no valid samples')
            continue

        dic = {
            'content_path': Series(data=content_paths),
            'image_path': Series(data=save_path),
            'type_id': Series(data=type_labels),
            'label': Series(data=labels),
            'frame_len': Series(data=frame_counts),
            'frame_seq': Series(data=frame_seq_counts)
        }

        pd.DataFrame(dic).to_csv(
            os.path.join(folder_path, f'{source}.csv'),
            encoding='utf-8', index=False)

        all_labels.extend(labels)
        all_save_path.extend(save_path)
        all_frame_counts.extend(frame_counts)
        all_frame_seq_counts.extend(frame_seq_counts)
        all_content_paths.extend(content_paths)
        all_type_labels.extend(type_labels)

    dic = {
        'content_path': Series(data=all_content_paths),
        'image_path': Series(data=all_save_path),
        'type_id': Series(data=all_type_labels),
        'label': Series(data=all_labels),
        'frame_len': Series(data=all_frame_counts),
        'frame_seq': Series(data=all_frame_seq_counts)
    }

    pd.DataFrame(dic).to_csv(
        os.path.join(folder_path, 'all.csv'),
        encoding='utf-8', index=False)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Scan a dataset folder of frame directories and generate CSV metadata"
    )
    parser.add_argument('--folder', dest='folder_path', required=True, help='Root directory to scan')
    parser.add_argument(
        '--label',
        dest='label',
        type=int,
        default=-1,
        help='Forced label (0=real, 1=fake); -1 to auto-infer from source folder name',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    generate_csv(args.folder_path, applied_label=args.label)
