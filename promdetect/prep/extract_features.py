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


# ANCILLARY FUNCTION
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


class Extractor(object):
    """
    Acoustic feature extractor.
    wav_file: Obligatory, path to a wav-file recording has to be supplied.
    nuclei: Processed DIRNDL annotation DataFrame on a syllable nucleus basis.
    gender: Gender of the speaker in the recording.

    The class specifies a large number of methods for the individual extraction of features for all nuclei in the provided DataFrame.
    Methods usually call Praat extraction functions that do the main work.
    """

    def __init__(self, wav_file, nuclei="", gender="f"):
        self.wav_file = wav_file
        self.snd_obj = pm.Sound(self.wav_file)
        self.nuclei = nuclei
        self.gender = gender

        # Different pitch ranges for female and male speakers
        if gender == "f":
            self.__pitch_range = (75, 500)
        else:
            self.__pitch_range = (50, 300)

    # EXTRACTION FUNCTIONS
    def calc_pitch_parts(self):
        """
        Create pitch objects for each of the nucleus sound objects extracted by `extract_parts()`
        """

        # Only run on nuclei with duration > 60 ms
        nuclei_filtered = self.nuclei[(self.nuclei["end"] - self.nuclei["start_est"]) >= 0.06].copy()

        nuclei_filtered["part_pitch"] = np.array(
            [
                row.part_obj.to_pitch_cc(
                    pitch_floor=self.__pitch_range[0],
                    pitch_ceiling=self.__pitch_range[1],
                )
                for row in nuclei_filtered.itertuples()
            ]
        )

        # Consolidate DataFrames
        self.nuclei["part_pitch"] = nuclei_filtered["part_pitch"]

    def get_rms(self):
        """
        Extract the RMS value for syllable nuclei.
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
        Extract the duration of syllable nuclei, normalized to the rest of their intonation phrase
        """

        check_input_df(self.nuclei, ["start_est", "end", "ip_start", "ip_end"])

        self.nuclei["duration"] = pd.Series(
            self.nuclei["end"] - self.nuclei["start_est"], dtype="float64"
        )

        if list(self.nuclei["duration"]) != []:

            # Group IPs and get each one's mean duration
            ip_mean = (
                self.nuclei[["ip_start", "ip_end", "duration"]]
                .groupby(["ip_start", "ip_end"], as_index=False)
                .mean()
            )
            ip_mean = ip_mean.rename(columns={"duration": "mean_ip_dur"})

            # Consolidate IP mean and nucleus DataFrames, so each nucleus row has info on the corresponding IP's mean duration
            durs_df = pd.merge(
                self.nuclei, ip_mean, how="left", on=["ip_start", "ip_end"]
            )

            normed_durs = (durs_df["duration"] / durs_df["mean_ip_dur"]).to_numpy()
        else:
            normed_durs = np.empty([])

        return normed_durs

    def get_pitch_slope(self):
        """
        Calculate the pitch slope, without octave jumps, across each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        # Requires independent sound and pitch contour slices for each nucleus.
        # Create if not existing.
        if "part_obj" not in self.nuclei.columns:
            self.extract_parts()
            self.calc_pitch_parts()
        else:
            if "part_pitch" not in self.nuclei.columns:
                self.calc_pitch_parts()
            else:
                pass

        nuclei_filtered = self.nuclei[self.nuclei["part_pitch"].notna()].copy()

        nuclei_filtered["pitch_slope"] = np.array(
            [
                part_pitch.get_slope_without_octave_jumps()
                for part_pitch in nuclei_filtered["part_pitch"]
            ]
        )

        return nuclei_filtered["pitch_slope"]

    def get_min_intensity_nuclei(self):
        """
        Extract the minimum intensity value in each syllable nucleus
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

    def get_max_intensity_nuclei(self):
        """
        Extract the maximum intensity value in each syllable nucleus
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

    def get_mean_intensity_nuclei(self):
        """
        Extract the mean intensity value for each nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["intens_mean"] = np.array(
            [
                praat.call(self.int_obj, "Get mean", row.start_est, row.end, "energy")
                for row in nuclei_filtered.itertuples()
            ]
        )

        return nuclei_filtered["intens_mean"]

    def get_intensity_std_nuclei(self):
        """
        Extract the standard deviation for intensity values across each syllable nucleus
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

    def get_min_intensity_pos(self):
        """
        Extract the relative position of the intensity minimum within the syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["intens_min_pos"] = np.array(
            [
                self.relative_position("minimum", "intensity", row.start_est, row.end)
                for row in nuclei_filtered.itertuples()
            ]
        )

        return nuclei_filtered["intens_min_pos"]

    def get_max_intensity_pos(self):
        """
        Extract the relative position of the intensity maximum within the syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["intens_max_pos"] = np.array(
            [
                self.relative_position("maximum", "intensity", row.start_est, row.end)
                for row in nuclei_filtered.itertuples()
            ]
        )

        return nuclei_filtered["intens_max_pos"]

    def get_intensity_ip(self):
        """
        Extract the mean intensity value for each intonation phrase
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
        Extract the F0 peak value in each syllable nucleus
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

        # Add to main DataFrame for other functions to use
        self.nuclei["f0_max"] = f0_max

        return f0_max

    def get_f0_min_nuclei(self):
        """
        Extract the minimal F0 value from each syllable nucleus
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

        # Add to main DataFrame for other functions to use
        self.nuclei["f0_min"] = f0_min

        return f0_min

    def get_f0_mean_nuclei(self):
        """
        Extract the mean F0 value for each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["f0_mean"] = [
            praat.call(self.pitch_obj, "Get mean", row.start_est, row.end, "Hertz")
            for row in nuclei_filtered.itertuples()
        ]

        f0_mean = nuclei_filtered["f0_mean"]

        return f0_mean

    def get_f0_range_nuclei(self):
        """
        Extract the F0 range (max - min) for each syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end", "f0_min", "f0_max"])

        f0_range = (self.nuclei["f0_max"] - self.nuclei["f0_min"]).to_numpy()

        return f0_range

    def get_f0_std_nuclei(self):
        """
        Extract the minimal F0 value from each syllable nucleus
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

        # Add to main DataFrame for other functions to use
        self.nuclei["f0_std"] = f0_std

        return f0_std

    def get_f0_min_pos(self):
        """
        Extract the relative position of the pitch minimum within a syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["f0_min_pos"] = [
            self.relative_position("minimum", "pitch", row.start_est, row.end)
            for row in nuclei_filtered.itertuples()
        ]

        return nuclei_filtered["f0_min_pos"]

    def get_f0_max_pos(self):
        """
        Extract the relative position of the pitch maximum within a syllable nucleus
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        nuclei_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        nuclei_filtered["f0_max_pos"] = [
            self.relative_position("maximum", "pitch", row.start_est, row.end)
            for row in nuclei_filtered.itertuples()
        ]

        return nuclei_filtered["f0_max_pos"]

    def get_excursion(self, level=""):
        """
        Extract the pitch excursion with normalization on either the "word" level or the intonation phrase ("ip") level
        """

        if level == "word":
            check_input_df(self.nuclei, ["word_start", "word_end", "f0_max"])

            timestamps_filtered = self.nuclei[
                (self.nuclei["word_start"].notna()) & (self.nuclei["word_end"].notna())
            ].copy()

            # Calculate 10th percentile of the pitch contour during nucleus
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

            # Calculate 10th percentile of the pitch contour during nucleus
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

        # Calculate excursion: 12 * log2(F0_max/F0_10%)
        excursions = np.array(12 * np.log2(norm_df["f0_max"] / norm_df["f0_q10"]))

        return excursions

    def get_spectral_tilt_mean(self):
        """
        Calculate the spectral tilt over the timespan of the syllable nucleus.
        Spectral tilt definition: Mean value of the first Mel-frequency cepstral coefficient (C1).
        Values are extracted from Praat MFCC objects.
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        # Requires independent sound slices for each nucleus.
        # Create if not existing.
        if "part_obj" not in self.nuclei.columns:
            self.extract_parts()

        timestamps_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        def calc_tilt(
            part_obj, start, end
        ):  # ancillary function to calculcate spectral tilt as mean C1 value over syllable nucleus

            # nucleus length needs to be at least 30 ms for analysis (2 * analysis frame length)
            if (end - start) > 0.03:
                nucl_mfcc = part_obj.to_mfcc(
                    number_of_coefficients=1, window_length=0.01
                )
                # C1 is second element in MFCC array
                nucl_tilt = np.mean(nucl_mfcc.to_array()[1])

            else:
                nucl_tilt = np.nan
            return nucl_tilt

        timestamps_filtered["tilt_mean"] = [
            calc_tilt(row.part_obj, row.start_est, row.end)
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

        # Requires independent sound slices for each nucleus.
        # Create if not existing.
        if "part_obj" not in self.nuclei.columns:
            self.extract_parts()

        timestamps_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        def calc_tilt_range(
            part_obj, start, end
        ):  # ancillary function to calculcate spectral tilt as mean C1 value over syllable nucleus

            # Nucleus length needs to be at least 30 ms for analysis (2 * analysis frame length)
            if (end - start) > 0.03:
                nucl_mfcc = part_obj.to_mfcc(
                    number_of_coefficients=1, window_length=0.01
                ).to_array()[
                    1
                ]  # C1 is second element in MFCC array
                nucl_tilt = max(nucl_mfcc) - min(nucl_mfcc)

            else:
                nucl_tilt = np.nan

            return nucl_tilt

        timestamps_filtered["tilt_range"] = [
            calc_tilt_range(row.part_obj, row.start_est, row.end)
            for row in timestamps_filtered.itertuples()
        ]

        tilt_range = timestamps_filtered["tilt_range"]

        return tilt_range

    def get_min_spectral_tilt(self):
        """
        Calculate the spectral tilt minimum during each syllable nucleus.
        Values are extracted from Praat MFCC objects.
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        # Requires independent sound slices for each nucleus.
        # Create if not existing.
        if "part_obj" not in self.nuclei.columns:
            self.extract_parts()

        timestamps_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        def calc_min_tilt(
            part_obj, start, end
        ):  # ancillary function to calculcate spectral tilt as minimum C1 value over syllable nucleus

            # Nucleus length needs to be at least 30 ms for analysis (2 * analysis frame length)
            if (end - start) > 0.03:
                nucl_mfcc = part_obj.to_mfcc(
                    number_of_coefficients=1, window_length=0.01
                ).to_array()[
                    1
                ]  # C1 is second element in MFCC array
                nucl_tilt = min(nucl_mfcc)

            else:
                nucl_tilt = np.nan

            return nucl_tilt

        timestamps_filtered["min_tilt"] = [
            calc_min_tilt(row.part_obj, row.start_est, row.end)
            for row in timestamps_filtered.itertuples()
        ]

        min_tilt = timestamps_filtered["min_tilt"]

        return min_tilt

    def get_max_spectral_tilt(self):
        """
        Calculate the spectral tilt maximum during each syllable nucleus.
        Values are extracted from Praat MFCC objects.
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        # Requires independent sound slices for each nucleus.
        # Create if not existing.
        if "part_obj" not in self.nuclei.columns:
            self.extract_parts()

        timestamps_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        def calc_max_tilt(
            part_obj, start, end
        ):  # ancillary function to calculcate spectral tilt as maximum C1 value over syllable nucleus

            # Nucleus length needs to be at least 30 ms for analysis (2 * analysis frame length)
            if (end - start) > 0.03:
                nucl_mfcc = part_obj.to_mfcc(
                    number_of_coefficients=1, window_length=0.01
                ).to_array()[
                    1
                ]  # C1 is second element in MFCC array
                nucl_tilt = max(nucl_mfcc)

            else:
                nucl_tilt = np.nan

            return nucl_tilt

        timestamps_filtered["max_tilt"] = [
            calc_max_tilt(row.part_obj, row.start_est, row.end)
            for row in timestamps_filtered.itertuples()
        ]

        max_tilt = timestamps_filtered["max_tilt"]

        return max_tilt

    def get_spectral_cog(self):
        """
        Extract the spectral center of gravity (CoG)
        """
        check_input_df(self.nuclei, ["start_est", "end"])

        # Requires independent sound slices for each nucleus.
        # Create if not existing.
        if "part_obj" not in self.nuclei.columns:
            self.extract_parts()

        timestamps_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        # Requires spectrum object for each nucleus slice
        timestamps_filtered["spec_part"] = [
            row.part_obj.to_spectrum() for row in timestamps_filtered.itertuples()
        ]

        timestamps_filtered["cog"] = [
            row.spec_part.get_center_of_gravity()
            for row in timestamps_filtered.itertuples()
        ]

        cog = timestamps_filtered["cog"]

        return cog

    def get_h1_h2(self):
        """
        Extract the H1-H2 value according to the calculation in Mooshammer (2010).
        """

        check_input_df(self.nuclei, ["start_est", "end"])

        # Requires independent sound and pitch contour slices for each nucleus.
        # Create if not existing.
        if "part_obj" not in self.nuclei.columns:
            self.extract_parts()
            self.calc_pitch_parts()
        else:
            if "part_pitch" not in self.nuclei.columns:
                self.calc_pitch_parts()
            else:
                pass

        timestamps_filtered = self.nuclei[
            (self.nuclei["start_est"].notna()) & (self.nuclei["end"].notna())
        ].copy()

        def calc_h1_h2(row):
            # Calculate bounds for more accurate Pitch object
            q25 = 0.75 * praat.call(
                self.pitch_obj, "Get quantile", row.start_est, row.end, 0.25, "Hertz"
            )
            q75 = 2.5 * praat.call(
                self.pitch_obj, "Get quantile", row.start_est, row.end, 0.75, "Hertz"
            )
            try:
                pitch_part = row.part_obj.to_pitch_cc(
                    pitch_floor=q25, pitch_ceiling=q75
                )

                # Get H1 (F0) and H2 frequencies, calculate their bandwidths.
                h1_freq = praat.call(pitch_part, "Get mean", 0, 0, "Hertz")
                h2_freq = h1_freq * 2

                h1_bw = 80 + 120 * h1_freq / 5_000
                h2_bw = 80 + 120 * h2_freq / 5_000

                # Filter sound signal for area around H1 and H2
                h1_filt_snd = praat.call(
                    row.part_obj, "Filter (one formant)", h1_freq, h1_bw
                )
                h2_filt_snd = praat.call(
                    row.part_obj, "Filter (one formant)", h2_freq, h2_bw
                )

                # Get intensity of filter bands
                h1 = praat.call(h1_filt_snd, "Get intensity (dB)")
                h2 = praat.call(h2_filt_snd, "Get intensity (dB)")

                return h1 - h2

            # Some bug in Parselmouth or Praat causes the calculation to fail sometimes, this is unfixable for now, so the function will just return NaN
            # TODO: investigate fix
            except Exception:
                return np.nan

        timestamps_filtered["h1_h2"] = [
            calc_h1_h2(row) for row in timestamps_filtered.itertuples()
        ]

        h1_h2 = timestamps_filtered["h1_h2"]

        return h1_h2

    # ANCILLARY FUNCTIONS
    def relative_position(self, extremum, type, start, end):
        """
        Calculate the relative position of either a maximum or minimum value within a timespan delimited by start and end timestamps
        extremum: one of "maximum" and "minimum"
        type: one of "pitch" and "intensity"
        """

        base = extremum_at = None

        if type == "pitch":
            base = self.pitch_obj
        elif type == "intensity":
            base = self.int_obj

        if type == "pitch":
            extremum_at = praat.call(
                base, f"Get time of {extremum}", start, end, "Hertz", "None"
            )
        elif type == "intensity":
            extremum_at = praat.call(
                base, f"Get time of {extremum}", start, end, "None"
            )

        time_passed = extremum_at - start
        relative_pos = time_passed / (end - start)

        return relative_pos

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
