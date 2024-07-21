import csv
from pathlib import Path
from lhotse import RecordingSet, SupervisionSet, Recording, SupervisionSegment, AudioSource

def prepare_basque_manifest(tsv_path, output_dir):

    recordings = list() # List to store Recording objects
    supervisions = list() # List to store SupervisionSegment objects
    
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for idx, row in enumerate(reader):

            # Generate a unique ID using the index
            unique_id = f"{Path(row['audio_filepath']).stem}_{idx}"
            
            # Create a Recording object for actual row entry
            recording = Recording(
                id=unique_id,
                sources=[AudioSource(type='file', channels=[0], source=row['audio_filepath'])],
                sampling_rate=16000,  # Adjust if different
                num_samples=int(float(row['duration']) * 16000),
                duration=float(row['duration'])
            )

            recordings.append(recording)
            
            # Create a SupervisionSegment object for actual row entry
            supervision = SupervisionSegment(
                id=unique_id,
                recording_id=unique_id,
                start=0,
                duration=float(row['duration']),
                text=row['text']
            )

            supervisions.append(supervision)
    
    # * Creates RecordingSet and SupervisionSet from the individual objects.
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # * Save manifests in the specified output directory
    recording_set.to_file(output_dir / 'recordings.jsonl.gz')
    supervision_set.to_file(output_dir / 'supervisions.jsonl.gz')

if __name__ == '__main__':

    train_tsv_path = "/mnt/ahogpu_ldisk2/adriang/icefall_own/egs/basque1/ASR/data/metadata/train.tsv"
    dev_tsv_path = "/mnt/ahogpu_ldisk2/adriang/icefall_own/egs/basque1/ASR/data/metadata/dev.tsv"
    test_cv_tsv_path = "/mnt/ahogpu_ldisk2/adriang/icefall_own/egs/basque1/ASR/data/metadata/test_cv.tsv"

    path_list = [train_tsv_path, dev_tsv_path, test_cv_tsv_path]
    splits = ['train', 'dev', 'test_cv']    

    for n, path in enumerate(path_list):
        prepare_basque_manifest(path, f'data/out/{splits[n]}')  # TSV files, output directory
