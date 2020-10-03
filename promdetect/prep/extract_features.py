"""
This script contains all preparatory extraction functions for the acoustic parameters that will later serve as the input for the neural network.
Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`numpy` to iterate with high performance
`pandas` to manage data
"""

import numpy as np
import pandas as pd
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


# EXTRACTION FUNCTIONS
def get_rms(snd_obj, nuclei):
    """
    This function extracts the RMS value for syllable nuclei.
    """

    check_input_df(nuclei, ["start_est", "end"])

    rms_vals = np.array(
        [
            snd_obj.get_rms(from_time=row.start_est, to_time=row.end)
            for row in nuclei.itertuples()
        ]
    )

    return rms_vals


def get_duration_normed(nuclei):
    """
    This function extracts the duration of syllable nuclei, normalized to the rest of their intonation phrase
    """

    check_input_df(nuclei, ["start_est", "end", "ip_start", "ip_end"])

    nuclei["duration"] = nuclei["end"] - nuclei["start_est"]

    ip_mean = nuclei.groupby(["ip_start", "ip_end"], as_index=False).mean()
    ip_mean = ip_mean[["ip_start", "ip_end", "duration"]]
    ip_mean = ip_mean.rename(columns={"duration": "mean_ip_dur"})

    durs_df = pd.merge(nuclei, ip_mean, how="left", on=["ip_start", "ip_end"])

    normed_durs = (durs_df["duration"] / durs_df["mean_ip_dur"]).to_numpy()

    return normed_durs


def get_intensity_nuclei(int_obj, nuclei):
    """
    This function extracts the maximum intensity value in each syllable nucleus
    """

    check_input_df(nuclei, ["start_est", "end"])

    intens_max = np.array(
        [
            praat.call(int_obj, "Get maximum", row.start_est, row.end, "None")
            for row in nuclei.itertuples()
        ]
    )

    return intens_max


def get_intensity_ip(int_obj, ip):
    """
    This function extracts the mean intensity value for each intonation phrase
    """

    check_input_df(ip, ["ip_start", "ip_end"])

    intens_avg = np.array(
        [
            praat.call(int_obj, "Get mean", row.ip_start, row.ip_end, "energy")
            for row in ip.itertuples()
        ]
    )

    return intens_avg


def get_f0_nuclei(pitch_obj, nuclei):
    """
    This function extracts the F0 peak value in each syllable nucleus
    """

    check_input_df(nuclei, ["start_est", "end"])

    f0_max = np.array(
        [
            praat.call(
                pitch_obj, "Get maximum", row.start_est, row.end, "Hertz", "None"
            )
            for row in nuclei.itertuples()
        ]
    )

    return f0_max


def get_excursion(pitch_obj, nuclei, level):
    """
    This function extracts the pitch excursion with normalization on either the "word" level or the intonation phrase ("ip") level
    """

    if level == "word":
        check_input_df(nuclei, ["word_start", "word_end", "f0_max"])

        timestamps = nuclei[["word_start", "word_end"]].drop_duplicates()

        timestamps["f0_q10"] = [
            praat.call(
                pitch_obj, "Get quantile", row.word_start, row.word_end, 0.1, "Hertz"
            )
            for row in timestamps.itertuples()
        ]

        norm_df = pd.merge(
            nuclei, timestamps, on=["word_start", "word_end"], how="left"
        )

    elif level == "ip":
        check_input_df(nuclei, ["ip_start", "ip_end", "f0_max"])

        timestamps = nuclei[["ip_start", "ip_end"]].drop_duplicates()

        timestamps["f0_q10"] = [
            praat.call(
                pitch_obj, "Get quantile", row.ip_start, row.ip_end, 0.1, "Hertz"
            )
            for row in timestamps.itertuples()
        ]

        norm_df = pd.merge(nuclei, timestamps, on=["ip_start", "ip_end"], how="left")

    else:
        raise ValueError("Argument 'level' must be one of ['word', 'ip']")

    excursions = np.array(12 * np.log2(norm_df["f0_max"] / norm_df["f0_q10"]))

    return excursions
