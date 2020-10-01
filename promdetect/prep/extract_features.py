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

    if all(col in nuclei.columns for col in ["start_est", "end"]):
        pass
    else:
        raise ValueError(
            "Input must be of type DataFrame and contain 'start_est' and 'end' columns."
        )

    rms_vals = np.array(
        [
            snd_obj.get_rms(from_time=row.start_est, to_time=row.end)
            for row in nuclei.itertuples()
        ]
    )

    return rms_vals
