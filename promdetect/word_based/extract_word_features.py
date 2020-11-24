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
        - minimum spectral tilt (MFCC-C1)
        - maximum spectral tilt (MFCC-C1)
        - mean spectral tilt (MFCC-C1)
        - spectral tilt range (MFCC-C1)
        - spectral centre of gravity
        - H1-H2
        """

        to_add = pd.DataFrame(
            columns=["tilt_min", "tilt_max", "tilt_mean", "tilt_range", "cog", "h1_h2"]
        )

        self.features = pd.concat([self.features, to_add])

        self.features_has_crit = self.features.copy().loc[
            (self.features["start"].notna())
            & (self.features["end"].notna())
            & (self.features["label"] != "<P>")
        ]

        if not hasattr(self, "pitch_obj"):
            self.pitch_obj = self.snd_obj.to_pitch_cc(
                pitch_floor=self.__pitch_range[0], pitch_ceiling=self.__pitch_range[1]
            )

        if "snd_part" not in self.features_has_crit.columns:
            self.features_has_crit["snd_part"] = [  # 10ms padding
                self.snd_obj.extract_part(
                    from_time=row.start - 0.01, to_time=row.end + 0.01
                )
                for row in self.features_has_crit.itertuples()
            ]

        self.features_has_crit["mfcc_part"] = [
            row.snd_part.to_mfcc(number_of_coefficients=1).to_array()[1]
            for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["spec_part"] = [
            row.snd_part.to_spectrum() for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["tilt_min"] = [
            np.min(row.mfcc_part) for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["tilt_max"] = [
            np.max(row.mfcc_part) for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["tilt_mean"] = [
            np.mean(row.mfcc_part) for row in self.features_has_crit.itertuples()
        ]

        self.features_has_crit["tilt_range"] = (
            self.features_has_crit["tilt_max"] - self.features_has_crit["tilt_min"]
        )

        self.features_has_crit["cog"] = [
            row.spec_part.get_center_of_gravity()
            for row in self.features_has_crit.itertuples()
        ]

        for row in self.features_has_crit.itertuples():
            q25 = 0.75 * praat.call(
                self.pitch_obj, "Get quantile", row.start, row.end, 0.25, "Hertz"
            )
            q75 = 2.5 * praat.call(
                self.pitch_obj, "Get quantile", row.start, row.end, 0.75, "Hertz"
            )
            try:
                pitch_part = row.snd_part.to_pitch_cc(
                    pitch_floor=q25, pitch_ceiling=q75
                )

                h1_freq = praat.call(pitch_part, "Get mean", 0, 0, "Hertz")
                h2_freq = h1_freq * 2

                h1_bw = 80 + 120 * h1_freq / 5_000
                h2_bw = 80 + 120 * h2_freq / 5_000

                h1_filt_snd = praat.call(
                    row.snd_part, "Filter (one formant)", h1_freq, h1_bw
                )
                h2_filt_snd = praat.call(
                    row.snd_part, "Filter (one formant)", h2_freq, h2_bw
                )

                h1 = praat.call(h1_filt_snd, "Get intensity (dB)")
                h2 = praat.call(h2_filt_snd, "Get intensity (dB)")

                self.features_has_crit.loc[row.Index, "h1_h2"] = h1 - h2

            except Exception:
                continue

        self.features.loc[
            (self.features["start"].notna())
            & (self.features["end"].notna())
            & (self.features["label"] != "<P>")
        ] = self.features_has_crit
