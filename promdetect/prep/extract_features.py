"""
This script contains all preparatory extraction functions for the acoustic parameters that will later serve as the input for the neural network.
Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`numpy` to iterate with high performance
`pandas` to manage data
"""

import numpy as np


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


# ANCILLARY FUNCTIONS
def check_input_df(input_df, expected_cols):

    if all(col in input_df.columns for col in expected_cols):
        pass
    else:
        raise ValueError(
            "Input must be of type DataFrame and contain the following columns: {}".format(
                ", ".join(expected_cols)
            )
        )
