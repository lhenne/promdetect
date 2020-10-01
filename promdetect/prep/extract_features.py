"""
This script contains all preparatory extraction functions for the acoustic parameters that will later serve as the input for the neural network.
Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`numpy` to iterate with high performance
`pandas` to manage data
"""

import numpy as np
import pandas as pd


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
