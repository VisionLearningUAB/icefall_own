from lhotse import CutSet, validate_recordings_and_supervisions

def validate_basque_data(manifest_dir):
    cuts = CutSet.from_file(f'{manifest_dir}/cuts.jsonl.gz')
    validate_recordings_and_supervisions(cuts)
    print(f"Cuts statistics:\n{cuts.describe()}")

if __name__ == '__main__':
    for split in ['train', 'dev', 'test_cv']:
        print(f"Validating {split} split:")
        validate_basque_data(f'data/fbank/{split}')
