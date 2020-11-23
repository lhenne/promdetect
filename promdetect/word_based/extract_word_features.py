import pandas as pd
import numpy as np
import parselmouth as pm


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
        - minimum intensity
        - maximum intensity
        - mean intensity
        - intensity std
        - minimum intensity relative position
        - maximum intensity relative position
        """

    def get_pitch_features(self):
        """
        Get pitch features:
        - minimum pitch
        - maximum pitch
        - mean pitch
        - pitch slope
        - pitch excursion
        - minimum pitch relative position
        - maximum pitch relative position
        """

    def get_spectral_features(self):
        """
        Get spectral features:
        - mean spectral tilt (C1)
        - spectral tilt range (C1)
        - spectral centre of gravity
        """
