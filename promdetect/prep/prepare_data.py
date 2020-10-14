"""
Import necessary packages:
`pandas` to manage data
`parselmouth` to send commands to the Praat phonetics software
"""

from pathlib import Path
import json
from promdetect.prep import process_annotations, find_syllable_nuclei, extract_features

"""
The functions in this module reformat the data from the DIRNDL corpus in order
to later be able to make use of them for the acoustic analysis.

This file coordinates and collects data by running functions defined in some of
the other files in this directory.
"""


class FeatureSet:
    def __init__(self, config, recording):
        with open(config, "r") as cfg:
            self.config = json.load(cfg)
        self.recording = recording
        self.wav_file = str(
            Path(self.config["directory"]).joinpath(f"{self.recording}.wav")
        )
        self.speaker = process_annotations.AnnotationReader(
            self.recording
        ).get_speaker_info()

    def run_config(self):
        self.accents = self.collect_annotations("accents")
        self.phones = self.collect_annotations("phones")
        self.tones = self.collect_annotations("tones")
        self.words = self.collect_annotations("words")

        if self.config["find_nuclei"]:
            points = find_syllable_nuclei.get_nucleus_points(self.wav_file)
            self.nuclei = find_syllable_nuclei.assign_points_labels(
                points, self.phones, self.words, self.tones
            )

        to_extract = [
            func for func, to_run in self.config["features"].items() if to_run
        ]  # compile list of functions that should be run according to the config

        if to_extract:
            extractor = extract_features.Extractor(
                self.wav_file, self.nuclei, self.tones, self.speaker[0], self.speaker[1]
            )
            features = {}

            for func_to_run in to_extract:
                self.call_function(extractor, features, func_to_run)

            return features

        else:
            pass

    def call_function(self, extractor, features_dict, func_to_run):
        if func_to_run == "excursion":
            levels = self.config["features_input"]["excursion"]
            if not hasattr(extractor, "pitch_obj"):
                extractor.calc_pitch()

            if "f0_max" not in self.nuclei.columns:
                self.nuclei["f0_max"] = extractor.get_f0_nuclei()

            for level in levels:
                key = f"excursion_{level}"
                features_dict[key] = extractor.get_excursion(level)

        else:
            if func_to_run in ["intensity_nuclei", "intensity_ip"]:
                if not hasattr(extractor, "intensity_obj"):
                    extractor.calc_intensity()

                features_dict[func_to_run] = eval(f"extractor.get_{func_to_run}()")

            elif func_to_run == "f0_nuclei":
                if not hasattr(extractor, "pitch_obj"):
                    extractor.calc_pitch()

                features_dict[func_to_run] = eval(f"extractor.get_{func_to_run}()")

            else:
                features_dict[func_to_run] = eval(f"extractor.get_{func_to_run}()")

    def collect_annotations(self, annotation_type):
        file = Path(self.config["directory"]).joinpath(
            f"{self.recording}.{annotation_type}"
        )
        return process_annotations.AnnotationReader(file).get_annotation_data()
