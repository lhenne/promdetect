from promdetect.word_based import extract_word_features
from promdetect.prep.process_annotations import AnnotationReader
import os

"""
Coordinate feature extraction with annotation processing steps, run for all recordings.
"""

with open(
    "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/list_recordings.txt",
    "r",
) as recordings:
    for recording in recordings:
        recording = recording.rstrip()

        if f"{recording}.csv" in os.listdir(
            "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/features/word_based"
        ):
            continue
        else:
            pass

        wav_file = f"/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/{recording}.wav"

        annot_dir = (
            "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/dirndl/word_based"
        )
        words = f"{annot_dir}/{recording}_words.csv"
        tones = f"{annot_dir}/{recording}_tones.csv"
        gender = AnnotationReader(recording).get_speaker_info()[1]

        extractor = extract_word_features.WordLevelExtractor(
            wav_file, words, tones, gender="m"
        )
        extractor.get_duration_features()
        extractor.get_intensity_features()
        extractor.get_pitch_features()
        extractor.get_spectral_features()

        extractor.features.to_csv(
            f"/home/lukas/Dokumente/Uni/ma_thesis/promdetect/data/features/word_based/{recording}.csv"
        )
