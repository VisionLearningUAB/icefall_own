
from pathlib import Path

train_csv_path = Path("/home/aholab/adriang/ahogpu_ldisk2_adriang/icefall_own/egs/basque1/ASR/local/train.csv")
dev_csv_path = Path("/home/aholab/adriang/ahogpu_ldisk2_adriang/icefall_own/egs/basque1/ASR/local/dev.csv")
test_cv_csv_path = Path("/home/aholab/adriang/ahogpu_ldisk2_adriang/icefall_own/egs/basque1/ASR/local/test_cv.csv")

import csv
from pathlib import Path
from lhotse import RecordingSet, SupervisionSet, Recording, SupervisionSegment

def prepare_basque_manifest(csv_path, output_dir):
    recordings = []
    supervisions = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            recording = Recording(
                id=row['id'],
                sources=[{'source': row['audio_filepath'], 'channels': [0]}],
                sampling_rate=16000,  # Adjust if different
                num_samples=int(float(row['duration']) * 16000),
                duration=float(row['duration'])
            )
            recordings.append(recording)
            
            supervision = SupervisionSegment(
                id=row['id'],
                recording_id=row['id'],
                start=0,
                duration=float(row['duration']),
                text=row['text']
            )
            supervisions.append(supervision)
    
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    recording_set.to_file(output_dir / 'recordings.jsonl.gz')
    supervision_set.to_file(output_dir / 'supervisions.jsonl.gz')

if __name__ == '__main__':

    train_csv_path = "/home/aholab/adriang/ahogpu_ldisk2_adriang/icefall_own/egs/basque1/ASR/local/train.csv"
    dev_csv_path = "/home/aholab/adriang/ahogpu_ldisk2_adriang/icefall_own/egs/basque1/ASR/local/dev.csv"
    test_cv_csv_path = "/home/aholab/adriang/ahogpu_ldisk2_adriang/icefall_own/egs/basque1/ASR/local/test_cv.csv"

    path_list = [train_csv_path, dev_csv_path, test_cv_csv_path]

    splits = ['train', 'dev', 'test_cv']    

    for n ,path in enumerate(path_list):
        prepare_basque_manifest(path, f'data/out/{splits[n]}') # CSV files, output directory
