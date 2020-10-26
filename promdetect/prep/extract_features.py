"""
This script contains all preparatory extraction functions for the acoustic parameters that will later serve as the input for the neural network.
Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`numpy` to iterate with high performance
`pandas` to manage data
"""

import numpy as np
import pandas as pd
import parselmouth as pm
from parselmouth import praat


# ANCILLARY FUNCTIONS
def check_input_df(input_df, expected_cols):
    """
    Check if the input DataFrame contains all the necessary columns for the extraction.
    """

    if all(col in input_df.columns for col in expected_cols):
        pass
    else:
        raise ValueError(
            "Input must be of type DataFrame and contain the following columns: {}".format(
                ", ".join(expected_cols)
            )
        )


# EXTRACTION


class Extractor(object):
    def __init__(self, wav_file, nuclei="", gender="f"):
        self.wav_file = wav_file
        self.snd_obj = pm.Sound(self.wav_file)
        self.nuclei = nuclei
        self.gender = gender

        if gender == "f":
            self.__pitch_range = (75, 500)
        else:
            self.__pitch_range = (50, 300)

    def calc_intensity(self):
        """
        Calculate Praat intensity object from sound object
        """

        self.int_obj = self.snd_obj.to_intensity(minimum_pitch=self.__pitch_range[0])

    def calc_pitch(self):
        """
        Calculate Praat pitch object from sound object
        """

        self.pitch_obj = self.snd_obj.to_pitch_cc(
            pitch_floor=self.__pitch_range[0], pitch_ceiling=self.__pitch_range[1]
        )

    def get_rms(self):
        """
        This function extracts the RMS value for syllable nuclei.
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        rms_vals = np.array(
            [
                self.snd_obj.get_rms(from_time=row.start_est, to_time=row.end)
                for row in self.nuclei.itertuples()
            ]
        )

        return rms_vals

    def get_duration_normed(self):
        """
        This function extracts the duration of syllable nuclei, normalized to the rest of their intonation phrase
        """

        check_input_df(self.nuclei, ["start_est", "end", "ip_start", "ip_end"])

        self.nuclei["duration"] = pd.Series(
            self.nuclei["end"] - self.nuclei["start_est"], dtype="float64"
        )

        if list(self.nuclei["duration"]) != []:
            ip_mean = (
                self.nuclei[["ip_start", "ip_end", "duration"]]
                .groupby(["ip_start", "ip_end"], as_index=False)
                .mean()
            )
            ip_mean = ip_mean.rename(columns={"duration": "mean_ip_dur"})

            durs_df = pd.merge(
                self.nuclei, ip_mean, how="left", on=["ip_start", "ip_end"]
            )

            normed_durs = (durs_df["duration"] / durs_df["mean_ip_dur"]).to_numpy()
        else:
            normed_durs = np.empty([])

        return normed_durs

    def get_intensity_nuclei(self):
        """
        This function extracts the maximum intensity value in each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["intens_max"] = np.array(
            [
                praat.call(self.int_obj, "Get maximum", row.start_est, row.end, "None")
                for row in nuclei_filtered.itertuples()
            ]
        )

        return nuclei_filtered["intens_max"]

    def get_intensity_ip(self):
        """
        This function extracts the mean intensity value for each intonation phrase
        """

        check_input_df(self.nuclei, ["ip_start", "ip_end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["ip_start"].notna()) & (self.nuclei["ip_end"].notna())
        ].copy()

        nuclei_filtered[
            "intens_avg"
        ] = np.array(  # TODO: this runs for each row, not for each IP
            [
                praat.call(self.int_obj, "Get mean", row.ip_start, row.ip_end, "energy")
                for row in nuclei_filtered.itertuples()
            ]
        )

        return nuclei_filtered["intens_avg"]

    def get_f0_max_nuclei(self):
        """
        This function extracts the F0 peak value in each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["f0_max"] = [
            praat.call(
                self.pitch_obj, "Get maximum", row.start_est, row.end, "Hertz", "None",
            )
            for row in nuclei_filtered.itertuples()
        ]

        f0_max = nuclei_filtered["f0_max"]

        self.nuclei["f0_max"] = f0_max

        return f0_max

    def get_excursion(self, level=""):
        """
        This function extracts the pitch excursion with normalization on either the "word" level or the intonation phrase ("ip") level
        """

        if level == "word":
            check_input_df(self.nuclei, ["word_start", "word_end", "f0_max"])

            timestamps_filtered = self.nuclei[
                (self.nuclei["word_start"].notna()) & (self.nuclei["word_end"].notna())
            ].copy()

            timestamps_filtered["f0_q10"] = [
                praat.call(
                    self.pitch_obj,
                    "Get quantile",
                    row.word_start,
                    row.word_end,
                    0.1,
                    "Hertz",
                )
                for row in timestamps_filtered.itertuples()
            ]

            norm_df = pd.merge(self.nuclei, timestamps_filtered, how="left")

        elif level == "ip":
            check_input_df(self.nuclei, ["ip_start", "ip_end", "f0_max"])

            nuclei_filtered = self.nuclei[
                (self.nuclei["ip_start"].notna()) & (self.nuclei["ip_end"].notna())
            ].copy()

            timestamps_filtered = nuclei_filtered[
                ["ip_start", "ip_end"]
            ].drop_duplicates()

            timestamps_filtered["f0_q10"] = [
                praat.call(
                    self.pitch_obj,
                    "Get quantile",
                    row.ip_start,
                    row.ip_end,
                    0.1,
                    "Hertz",
                )
                for row in timestamps_filtered.itertuples()
            ]

            norm_df = pd.merge(
                self.nuclei, timestamps_filtered, on=["ip_start", "ip_end"], how="left"
            )

        else:
            raise ValueError("Argument 'level' must be one of ['word', 'ip']")

        excursions = np.array(12 * np.log2(norm_df["f0_max"] / norm_df["f0_q10"]))

        return excursions
