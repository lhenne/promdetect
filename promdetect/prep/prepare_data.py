from pathlib import Path
import json
from glob import glob
from promdetect.prep import process_annotations, find_syllable_nuclei, extract_features

"""
The functions in this module reformat the data from the DIRNDL corpus in order
to later be able to make use of them for the acoustic analysis.

This file coordinates and collects data by running functions defined in some of
the other files in this directory.
"""

CFG_FILE = "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/promdetect/prep/config.json"
with open(CFG_FILE, "r") as cfg:
    CONFIG = json.load(cfg)


class FeatureSet:
    def __init__(self, config, recording):
        self.config = config
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
            self.nuclei_raw = find_syllable_nuclei.assign_points_labels(
                points, self.phones, self.words, self.tones, self.accents
            )

        self.nuclei = self.nuclei_raw.loc[
            self.nuclei_raw["phone"].notna()
        ].reset_index()

        to_extract = [
            func for func, to_run in self.config["features"].items() if to_run
        ]  # compile list of functions that should be run according to the config

        if to_extract:
            extractor = extract_features.Extractor(
                self.wav_file, self.nuclei, self.speaker[1]
            )
            features = self.nuclei.copy()

            # Will add extracted feature values to main feature DataFrame
            for func_to_run in to_extract:
                self.call_function(extractor, features, func_to_run)

            return features

        else:
            pass

    def call_function(self, extractor, features_df, func_to_run):
        """
        Call the extraction functions based on which are activated in config.json.
        Different functions have different prerequisites.
        """

        if func_to_run == "excursion":
            levels = self.config["features_input"]["excursion"]
            if not hasattr(extractor, "pitch_obj"):
                extractor.calc_pitch()

            if "f0_max" not in self.nuclei.columns:
                self.nuclei["f0_max"] = extractor.get_f0_max_nuclei()

            for level in levels:
                features_df[f"excursion_{level}"] = extractor.get_excursion(level)

        else:
            if func_to_run in [
                "max_intensity_nuclei",
                "min_intensity_nuclei",
                "mean_intensity_nuclei",
                "intensity_ip",
                "intensity_std_nuclei",
                "intensity_min_pos",
                "intensity_max_pos",
            ]:
                if not hasattr(extractor, "int_obj"):
                    extractor.calc_intensity()

                features_df[func_to_run] = eval(f"extractor.get_{func_to_run}()")

            elif func_to_run in [
                "f0_max_nuclei",
                "f0_min_nuclei",
                "f0_mean_nuclei",
                "pitch_slope",
                "f0_range_nuclei",
                "f0_std_nuclei",
                "f0_min_pos",
                "f0_max_pos",
                "h1_h2",
            ]:
                if not hasattr(extractor, "pitch_obj"):
                    extractor.calc_pitch()

                features_df[func_to_run] = eval(f"extractor.get_{func_to_run}()")

            else:
                features_df[func_to_run] = eval(f"extractor.get_{func_to_run}()")

    def collect_annotations(self, annotation_type):
        file = Path(self.config["directory"]).joinpath(
            f"{self.recording}.{annotation_type}"
        )
        return process_annotations.AnnotationReader(file).get_annotation_data()


if __name__ == "__main__":
    """
    Call for all recordings
    """
    directory = CONFIG["directory"]
    recordings = [Path(file).stem for file in glob(f"{directory}/*.wav")]

    out_dir = str(Path(input("Please enter output directory: ")).resolve()) + "/"

    # Check for already existing data and skip if found
    existing = [Path(file).stem for file in glob(f"{out_dir}*")]
    idx = 1
    recordings = [recording for recording in recordings if recording not in existing]

    for recording in recordings:
        feature_set = FeatureSet(CONFIG, recording)
        feature_data = feature_set.run_config()

        limit = len(recordings)

        with open(out_dir + recording, "w+") as out_file:
            feature_data.to_csv(out_file)

        print(f"Processed {idx} of {limit} recordings.", end="\r")
        idx += 1
