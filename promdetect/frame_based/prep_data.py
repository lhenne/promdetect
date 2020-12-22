from pathlib import Path
from glob import glob
from os import chdir
from json import load
from promdetect.frame_based.extract_frame_features import FrameLevelExtractor

# Import dictionary containing speaker gender for each recording
with open(
    "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/speakers.json", "r"
) as speaker_file:
    SPEAKERS = load(speaker_file)

INPUT_PATH = Path(input("Enter directory containing input WAV files: ")).resolve()
chdir(INPUT_PATH)

OUTPUT_PATH = Path(input("Enter output directory: ")).resolve()

# Go through all WAV files and extract frame-level features
# Dump results in a '.wav.frames' file each
# Output directory promdetect/data/features/frame_based

cur_rec = 1
for wav in glob("*.wav"):
    print(f"Processing {cur_rec} of 55 recordings.")
    cur_rec += 1

    recording = Path(wav).stem
    gender = SPEAKERS[recording]

    if Path(f"{OUTPUT_PATH}/{recording}.wav.frames").exists():
        print(f"Skipping recording {recording}")
        continue
    else:
        pass

    extractor = FrameLevelExtractor(wav, gender, OUTPUT_PATH)
    extractor.rms_extraction()
    extractor.loudness_extraction()
    extractor.zcr_extraction()
    extractor.hnr_extraction()
    extractor.write_features()
