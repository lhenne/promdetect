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

        self.features_has_crit = self.features.copy().loc[
            (self.features["start"].notna())
            & (self.features["end"].notna())
            & (self.features["label"] != "<P>")
        ]

        self.features_has_crit["int_rms"] = [
            praat.call(self.snd_obj, "Get root-mean-square", row.start, row.end)
            for row in self.features_has_crit.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features_has_crit["int_min"] = [
            praat.call(self.int_obj, "Get minimum", row.start, row.end, "None")
            for row in self.features_has_crit.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features_has_crit["int_max"] = [
            praat.call(self.int_obj, "Get maximum", row.start, row.end, "None")
            for row in self.features_has_crit.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features_has_crit["int_mean"] = [
            praat.call(self.int_obj, "Get mean", row.start, row.end, "energy")
            for row in self.features_has_crit.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features_has_crit["int_std"] = [
            praat.call(self.int_obj, "Get standard deviation", row.start, row.end)
            for row in self.features_has_crit.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features_has_crit["int_min_pos"] = [
            (
                praat.call(
                    self.int_obj, "Get time of minimum", row.start, row.end, "None"
                )
                - row.start
            )
            / (row.end - row.start)
            for row in self.features_has_crit.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features_has_crit["int_max_pos"] = [
            (
                praat.call(
                    self.int_obj, "Get time of maximum", row.start, row.end, "None"
                )
                - row.start
            )
            / (row.end - row.start)
            for row in self.features_has_crit.itertuples()
            if row.start == row.start and row.end == row.end and row.label != "<P>"
        ]

        self.features.loc[
            (self.features["start"].notna())
            & (self.features["end"].notna())
            & (self.features["label"] != "<P>")
        ] = self.features_has_crit

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
        to_add = pd.DataFrame(
            columns=[
                "f0_min",
                "f0_max",
                "f0_mean",
                "f0_std",
                "f0_slope",
                "f0_exc_ip",
                "f0_exc_utt",
                "f0_min_pos",
                "f0_max_pos",
            ]
        )

        self.features = pd.concat([self.features, to_add])

        self.pitch_obj = self.snd_obj.to_pitch_cc(
            pitch_floor=self.__pitch_range[0], pitch_ceiling=self.__pitch_range[1]
        )

        self.features_has_crit = self.features.copy().loc[  # TODO: Filter function to call in all `get` blocks
            (self.features["start"].notna())
            & (self.features["end"].notna())
            & (self.features["label"] != "<P>")
        ]

        self.features_has_crit["snd_part"] = [  # 10ms padding
            self.snd_obj.extract_part(
                from_time=row.start - 0.01, to_time=row.end + 0.01
            )
            for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["pitch_part"] = [
            row.snd_part.to_pitch_cc(
                pitch_floor=self.__pitch_range[0], pitch_ceiling=self.__pitch_range[1]
            )
            for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["f0_min"] = [
            praat.call(
                self.pitch_obj, "Get minimum", row.start, row.end, "Hertz", "None"
            )
            for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["f0_max"] = [
            praat.call(
                self.pitch_obj, "Get maximum", row.start, row.end, "Hertz", "None"
            )
            for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["f0_mean"] = [
            praat.call(self.pitch_obj, "Get mean", row.start, row.end, "Hertz")
            for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["f0_std"] = [
            praat.call(
                self.pitch_obj, "Get standard deviation", row.start, row.end, "Hertz"
            )
            for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["f0_slope"] = [
            row.pitch_part.get_slope_without_octave_jumps()
            for row in self.features_has_crit.itertuples()
        ]

        for ip in self.tones.itertuples():
            if ip.start == ip.start and ip.end == ip.end:
                f0_q10 = praat.call(
                    self.pitch_obj, "Get quantile", ip.start, ip.end, 0.1, "Hertz"
                )

                self.features_has_crit.loc[
                    (ip.start <= self.features_has_crit["start"])
                    & (self.features_has_crit["end"] <= ip.end),
                    "f0_exc_ip",
                ] = (
                    12
                    * np.log2(
                        self.features_has_crit.loc[
                            (ip.start <= self.features_has_crit["start"])
                            & (self.features_has_crit["end"] <= ip.end),
                            "f0_max",
                        ]
                        / f0_q10
                    )
                )

        bounds = (
            self.features.copy()
            .loc[
                (self.features["start"].notna())
                & (self.features["end"].notna())
                & (self.features["label"] == "<P>")
            ]
            .reset_index()
        )

        for row in bounds.itertuples():
            if row.start == row.start and row.end == row.end:

                if row.level_0 < len(bounds):
                    utt_start = row.end
                    try:
                        utt_end = bounds.loc[bounds.index == int(row.Index) + 1][
                            "start"
                        ].item()
                    except Exception:
                        continue
                else:
                    utt_start = row.start
                    utt_end = row.end

                f0_q10 = praat.call(
                    self.pitch_obj, "Get quantile", utt_start, utt_end, 0.1, "Hertz"
                )

                self.features_has_crit.loc[
                    (row.start <= self.features_has_crit["start"])
                    & (self.features_has_crit["end"] <= utt_end),
                    "f0_exc_utt",
                ] = (
                    12
                    * np.log2(
                        self.features_has_crit.loc[
                            (row.start <= self.features_has_crit["start"])
                            & (self.features_has_crit["end"] <= utt_end),
                            "f0_max",
                        ]
                        / f0_q10
                    )
                )

        self.features_has_crit["f0_min_pos"] = [
            (
                praat.call(
                    self.pitch_obj,
                    "Get time of minimum",
                    row.start,
                    row.end,
                    "Hertz",
                    "None",
                )
                - row.start
            )
            / (row.end - row.start)
            for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["f0_max_pos"] = [
            (
                praat.call(
                    self.pitch_obj,
                    "Get time of maximum",
                    row.start,
                    row.end,
                    "Hertz",
                    "None",
                )
                - row.start
            )
            / (row.end - row.start)
            for row in self.features_has_crit.itertuples()
        ]

        self.features.loc[
            (self.features["start"].notna())
            & (self.features["end"].notna())
            & (self.features["label"] != "<P>")
        ] = self.features_has_crit

    def get_spectral_features(self):
        """
        Get spectral features:
        - mean spectral tilt (C1)
        - spectral tilt range (C1)
        - spectral centre of gravity
        - H1-H2
        """
