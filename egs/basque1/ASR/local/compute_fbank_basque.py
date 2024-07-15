from lhotse import CutSet, Fbank
from lhotse.recipes import read_manifests_if_cached

def compute_fbank_basque(manifest_dir, output_dir):
    recordings, supervisions = read_manifests_if_cached(manifest_dir)
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    
    extractor = Fbank()
    cuts = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=output_dir
    )
    
    cuts.to_file(f'{output_dir}/cuts.jsonl.gz')

if __name__ == '__main__':
    for split in ['train', 'dev', 'test_cv']:
        compute_fbank_basque(f'data/manifests/{split}', f'data/fbank/{split}')
