import argparse
import os
import glob
import shutil
import h5py
import pandas as pd
import tqdm
from typing import List

from figaro.data.utils import download_extract_tar


def match_midi_and_annotations(
    lmd_clean_url: str,
    isophonics_url: str,
    output_path: str,
    artist_name: str = "The Beatles",
):
    lmd_clean_path = download_extract_tar(lmd_clean_url)
    isophonics_path = download_extract_tar(isophonics_url)

    # Get the list of Isophonics annotation files
    isophonics_annotations = glob.glob(os.path.join(isophonics_path, "chordlab", artist_name, "**", "*.lab"))
    clean_files = glob.glob(os.path.join(lmd_clean_path, artist_name, "*.mid"))

    # Create a dataframe with columns 'title' and 'path'
    # where path is the relative path to the clean MIDI file
    rows = []
    for clean_file in clean_files:
        _, raw_song_title = os.path.split(clean_file)
        song_title = raw_song_title.split(".")[0]

        rows.append({
            'title': song_title,
            'path': clean_file,
        })
    metadata = pd.DataFrame(rows)


    # Count the total number of annotation files to process
    num_files = len(isophonics_annotations)

    # Iterate over the Isophonics annotation files
    n_matched = 0
    with tqdm.tqdm(total=num_files, desc='Matching', unit='file') as pbar:
        for annotation_file in isophonics_annotations:
            _, raw_song_title = os.path.split(annotation_file)
            song_title = os.path.splitext(raw_song_title)[0].split("_-_")[-1].replace("_", " ")

            # Find the matching MIDI files in the metadata
            matched_files = metadata.loc[
                metadata["title"].str.lower() == song_title.lower()
            ]

            # If there are matched MIDI files, copy them to the output folder along with the annotations
            if not matched_files.empty:
                for _, row in matched_files.iterrows():
                    midi_file = row["path"]

                    # Create the output directory if it doesn't exist
                    os.makedirs(output_path, exist_ok=True)

                    # Copy the MIDI file and annotation file to the output folder
                    os.makedirs(os.path.join(output_path, artist_name, song_title), exist_ok=True)
                    shutil.copy2(midi_file, os.path.join(output_path, artist_name, song_title, os.path.basename(midi_file)))
                    shutil.copy2(annotation_file, os.path.join(output_path, artist_name, song_title, "chords.lab"))

                    # print(f"Matched: {artist_name} - {song_title}")
                n_matched += 1

            pbar.update(1)
    print(f"Matched {n_matched} files out of {num_files} total files.")


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Match MIDI files and annotations')
    parser.add_argument('--lmd-clean-url', type=str, default='http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz', help='URL of the LakhMIDI clean dataset files')
    parser.add_argument('--isophonics-url', type=str, default='http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz', help='URL of the Isophonics annotation file')
    parser.add_argument('--output-path', type=str, default='data_cache/isophonics_annotated', help='Path to the output folder')
    args = parser.parse_args()

    # Call the match_midi_and_annotations function with the parsed arguments
    match_midi_and_annotations(
        args.lmd_clean_url,
        args.isophonics_url,
        args.output_path
    )


if __name__ == "__main__":
    main()
