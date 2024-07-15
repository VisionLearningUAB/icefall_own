import os
from lhotse import CutSet, Fbank, RecordingSet, SupervisionSet

def compute_fbank_basque(manifest_dir, output_dir):
    # Load manifests
    recordings = RecordingSet.from_jsonl(f'{manifest_dir}/recordings.jsonl')
    supervisions = SupervisionSet.from_jsonl(f'{manifest_dir}/supervisions.jsonl')
    
    # Create CutSet
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    
    # Create Fbank extractor
    extractor = Fbank()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Compute and store features
    cuts = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=output_dir
    )
    
    # Save the cuts
    cuts.to_file(f'{output_dir}/cuts.jsonl.gz')

if __name__ == '__main__':
    for split in ['train', 'dev', 'test_cv']:
        compute_fbank_basque(f'data/out/{split}', f'data/fbank/{split}')
