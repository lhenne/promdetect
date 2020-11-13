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

    def extract_parts(self):
        """
        Extract separate Praat sound objects for each of the nuclei in `self.nuclei`.
        Add them to DataFrame.
        """

        self.nuclei["part_obj"] = [
            self.snd_obj.extract_part(from_time=row.start_est, to_time=row.end)
            for row in self.nuclei.itertuples()
        ]

    def calc_pitch_parts(self):
        """
        Create pitch objects for each of the nucleus sound objects extracted by `extract_parts()`
        """

        self.nuclei["part_pitch"] = np.array(
            [
                row.part_obj.to_pitch_cc(
                    pitch_floor=self.__pitch_range[0],
                    pitch_ceiling=self.__pitch_range[1],
                )
                for row in self.nuclei.itertuples()
            ]
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

    def get_pitch_slope(self):
        """
        This function calculates the pitch slope, without octave jumps, across each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        if "part_obj" not in self.nuclei.columns:
            self.extract_parts()
            self.calc_pitch_parts()
        else:
            if "part_pitch" not in self.nuclei.columns:
                self.calc_pitch_parts()
            else:
                pass

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["pitch_slope"] = np.array(
            [
                part_pitch.get_slope_without_octave_jumps()
                for part_pitch in self.nuclei["part_pitch"]
            ]
        )

        return nuclei_filtered["pitch_slope"]

    def get_min_intensity_nuclei(self):
        """
        This function extracts the minimum intensity value in each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["intens_min"] = np.array(
            [
                praat.call(self.int_obj, "Get minimum", row.start_est, row.end, "None")
                for row in nuclei_filtered.itertuples()
            ]
        )

        return nuclei_filtered["intens_min"]

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

    def get_intensity_std_nuclei(self):
        """
        This function extracts the standard deviation for intensity values across each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["intens_std"] = np.array(
            [
                praat.call(
                    self.int_obj, "Get standard deviation", row.start_est, row.end
                )
                for row in nuclei_filtered.itertuples()
            ]
        )

        return nuclei_filtered["intens_std"]

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

    def get_f0_min_nuclei(self):
        """
        This function extracts the minimal F0 value from each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["f0_min"] = [
            praat.call(
                self.pitch_obj, "Get minimum", row.start_est, row.end, "Hertz", "None",
            )
            for row in nuclei_filtered.itertuples()
        ]

        f0_min = nuclei_filtered["f0_min"]

        self.nuclei["f0_min"] = f0_min

        return f0_min

    def get_f0_range_nuclei(self):
        """
        This function extracts the F0 range (max - min) for each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end", "f0_min", "f0_max"])

        f0_range = (self.nuclei["f0_max"] - self.nuclei["f0_min"]).to_numpy()

        return f0_range

    def get_f0_std_nuclei(self):
        """
        This function extracts the minimal F0 value from each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["f0_std"] = [
            praat.call(
                self.pitch_obj,
                "Get standard deviation",
                row.start_est,
                row.end,
                "Hertz",
            )
            for row in nuclei_filtered.itertuples()
        ]

        f0_std = nuclei_filtered["f0_std"]

        self.nuclei["f0_std"] = f0_std

        return f0_std

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

    def get_spectral_tilt_mean(self):
        """
        Calculate the spectral tilt over the timespan of the syllable nucleus.
        Spectral tilt definition: Mean value of the first Mel-frequency cepstral coefficient (C1).
        Values are extracted from Praat MFCC objects.
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        timestamps_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        def calc_tilt(
            snd_obj, start, end
        ):  # ancillary function to calculcate spectral tilt as mean C1 value over syllable nucleus

            if (end - start) > 0.03:
                nucl_obj = snd_obj.extract_part(from_time=start, to_time=end)
                nucl_mfcc = nucl_obj.to_mfcc(
                    number_of_coefficients=1, window_length=0.01
                )
                nucl_tilt = np.mean(nucl_mfcc.to_array()[1])

            else:
                nucl_tilt = np.nan
            return nucl_tilt

        timestamps_filtered["tilt_mean"] = [
            calc_tilt(self.snd_obj, row.start_est, row.end)
            for row in timestamps_filtered.itertuples()
        ]

        tilt_mean = timestamps_filtered["tilt_mean"]

        return tilt_mean

    def get_spectral_tilt_range(self):
        """
        Calculate the spectral tilt range (max - min) over the timespan of the syllable nucleus.
        Values are extracted from Praat MFCC objects.
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        timestamps_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        def calc_tilt_range(
            snd_obj, start, end
        ):  # ancillary function to calculcate spectral tilt as mean C1 value over syllable nucleus

            if (end - start) > 0.03:
                nucl_obj = snd_obj.extract_part(from_time=start, to_time=end)
                nucl_mfcc = nucl_obj.to_mfcc(
                    number_of_coefficients=1, window_length=0.01
                ).to_array()[1]
                nucl_tilt = max(nucl_mfcc) - min(nucl_mfcc)

            else:
                nucl_tilt = np.nan

            return nucl_tilt

        timestamps_filtered["tilt_range"] = [
            calc_tilt_range(self.snd_obj, row.start_est, row.end)
            for row in timestamps_filtered.itertuples()
        ]

        tilt_range = timestamps_filtered["tilt_range"]

        return tilt_range
