import pandas as pd
import numpy as np
import parselmouth as pm
from parselmouth import praat


class WordLevelExtractor:
    def __init__(self, wav_file, words, tones, gender="f"):
        self.wav_file = wav_file
        self.words = pd.read_csv(words)
        self.tones = pd.read_csv(tones)
        self.snd_obj = pm.Sound(self.wav_file)
        self.gender = gender
        self.features = pd.DataFrame(self.words)

        if gender == "f":
            self.__pitch_range = (75, 500)
        else:
            self.__pitch_range = (50, 300)

    def get_duration_features(self):
        """
        Get duration features:
        - relative duration compared to IP mean
        """
        to_add = pd.DataFrame(columns=["dur", "dur_normed"])
        self.features = pd.concat([self.features, to_add])

        self.features["dur"] = self.words["end"] - self.words["start"]

        for row in self.tones.itertuples():
            if row.start == row.start and row.end == row.end:  # check if NaN
                ip_mean = np.mean(
                    self.features.loc[
                        (self.features["label"] != "<P>")
                        & (  # get normed duration if word during current IP, and word label is not indicating punctuation
                            row.start <= self.features["start"]
                        )
                        & (self.features["end"] <= row.end),
                        "dur",
                    ]
                )
                self.features.loc[
                    (self.features["label"] != "<P>")
                    & (row.start <= self.features["start"])
                    & (self.features["end"] <= row.end),
                    "dur_normed",
                ] = (
                    self.features.loc[
                        (self.features["label"] != "<P>")
                        & (row.start <= self.features["start"])
                        & (self.features["end"] <= row.end),
                        "dur",
                    ]
                    / ip_mean
                )

            else:
                pass

    def get_intensity_features(self):
        """
        Get intensity features:
        - rms
        - minimum intensity
        - maximum intensity
        - mean intensity
        - intensity std
        - minimum intensity relative position
        - maximum intensity relative position
        """

        to_add = pd.DataFrame(
            columns=[
                "int_rms",
                "int_min",
                "int_max",
                "int_mean",
                "int_std",
                "int_min_pos",
                "int_max_pos",
            ]
        )
        self.features = pd.concat([self.features, to_add])

        self.int_obj = self.snd_obj.to_intensity(minimum_pitch=self.__pitch_range[0])

        self.features["int_rms"] = [
            praat.call(self.snd_obj, "Get root-mean-square", row.start, row.end)
            for row in self.features.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features["int_min"] = [
            praat.call(self.int_obj, "Get minimum", row.start, row.end, "None")
            for row in self.features.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features["int_max"] = [
            praat.call(self.int_obj, "Get maximum", row.start, row.end, "None")
            for row in self.features.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features["int_mean"] = [
            praat.call(self.int_obj, "Get mean", row.start, row.end, "energy")
            for row in self.features.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features["int_std"] = [
            praat.call(self.int_obj, "Get standard deviation", row.start, row.end)
            for row in self.features.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features["int_min_pos"] = [
            (
                praat.call(
                    self.int_obj, "Get time of minimum", row.start, row.end, "None"
                )
                - row.start
            )
            / row.dur
            for row in self.features.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features["int_max_pos"] = [
            (
                praat.call(
                    self.int_obj, "Get time of maximum", row.start, row.end, "None"
                )
                - row.start
            )
            / row.dur
            for row in self.features.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

    def get_pitch_features(self):
        """
        Get pitch features:
        - minimum pitch
        - maximum pitch
        - mean pitch
        - pitch standard deviation
        - pitch slope
        - pitch excursion relative to IP
        - pitch excursion relative to utterance (sentence)
        - minimum pitch relative position
        - maximum pitch relative position
        """

    def get_spectral_features(self):
        """
        Get spectral features:
        - mean spectral tilt (C1)
        - spectral tilt range (C1)
        - spectral centre of gravity
        - H1-H2
        """
