"""
This script detects syllable nuclei in a sound file
and returns their timestamps.

It is loosely based on a Praat script by Nivja de Jong and Ton Wempe:
https://sites.google.com/site/speechrate/Home/praat-script-syllable-nuclei-v2

Their approach to voicing detection is switched out against making use of the additional information that phone- and word-level labels in the DIRNDL corpus offers.

The analysis is run on filtered versions of the input files, filtering is done
by means of an internal Praat function with custom parameters.


Import necessary packages:
`parselmouth` to send commands to the Praat phonetics software
`pandas` to manage and output data in a nice format
"""
from pandas import DataFrame
import parselmouth as pm
from parselmouth import praat

SAMPA_VOWELS = [  # Vowel symbols in SAMPA, to separate vowel timestamps from consonant timestamps
    "a:",
    "e:",
    "i:",
    "o:",
    "u:",
    "E:",
    "2:",
    "y:",
    "a",
    "E",
    "I",
    "O",
    "U",
    "Y",
    "9",
    "@",
    "6",
    "OY",
    "aI",
    "aU",
    "_6",
    "_9",
    "_2:",
]
EXTRALING_SOUNDS = [
    "[@]",
    "[t]",
    "[n]",
    "[f]",
    "[h]",
    "<P>",
]  # breathing sounds in the word-level DIRNDL annotation

SILENCE_THRESHOLD = -25  # threshold to separate silence and voice in dB
MIN_DIP_BETW_PEAKS = 2  # minimum dip between intensity peaks in dB


def get_nucleus_points(sound_file):
    """
    This function determines syllable nuclei in an input sound file.
    """

    snd_raw = pm.Sound(sound_file)

    # Clean parselmouth sound obj object by removing noise using a Praat function.
    # ["Remove noise"] from start [0] to end [0] of the Sound, work with overlapping windows with length [0.025] seconds, filter everything between frequencies [50] Hz to [10_000] Hz with [40] Hz smoothing factor using the ["Spectral subtraction"] noise reduction method.
    snd_filtered = praat.call(
        snd_raw, "Remove noise", 0, 0, 0.025, 75, 10_000, 40, "Spectral subtraction"
    )

    intensity_obj = snd_filtered.to_intensity(minimum_pitch=75)

    min_intensity = intensity_obj.get_minimum()
    max_intensity_99 = praat.call(
        intensity_obj, "Get quantile", 0, 0, 0.99
    )  # to remove extreme outliers (top 1%)

    threshold = max_intensity_99 + SILENCE_THRESHOLD

    if threshold < min_intensity:
        threshold = min_intensity  # in case the threshold was set lower than the silence level in the recording

    peak_cands = find_peak_cands(intensity_obj, threshold)

    valid_peaks = validate(intensity_obj, peak_cands)

    return [x[0] for x in valid_peaks]


# ANCILLARY FUNCTIONS


def assign_points_labels(nuclei, phones, words, tones, accents):
    """
    This function assigns phone and word labels, as well as phone boundaries and the corresponding durations.
    """

    phones_filtered = filter_labels(phones, "phones")
    words_filtered = filter_labels(words, "words")

    assigned_df = DataFrame(  # prepare empty DF to insert all the combined data into
        columns=[
            "nucl_time",
            "phone",
            "word",
            "bound_tone",
            "start_est",
            "end",
            "duration_est",
            "word_start",
            "word_end",
            "ip_start",
            "ip_end",
            "accent_time",
            "accent_label",
        ]
    )

    assigned_df["nucl_time"] = nuclei

    for row in assigned_df.itertuples():
        match_phones = phones_filtered.loc[
            (phones_filtered["start_est"] <= row.nucl_time)
            & (phones_filtered["end"] >= row.nucl_time)
        ].to_numpy()

        match_words = words_filtered.loc[
            (words_filtered["start_est"] <= row.nucl_time)
            & (words_filtered["end"] >= row.nucl_time)
        ]
        match_words = match_words.rename(
            columns={"start_est": "word_start", "end": "word_end"}
        ).to_numpy()

        match_tones = tones.loc[
            (tones["start_est"] <= row.nucl_time) & (tones["time"] >= row.nucl_time)
        ]
        match_tones = match_tones.rename(
            columns={"start_est": "ip_start", "time": "ip_end"}
        ).to_numpy()

        if match_phones.any():
            match_accents = accents.loc[
                (match_phones[0][2] <= accents["time"])
                & (match_phones[0][0] >= accents["time"])
            ].to_numpy()

            assigned_df.at[row.Index, ["end", "phone", "start_est"]] = match_phones[0]

            if match_accents.any():
                assigned_df.at[
                    row.Index, ["accent_time", "accent_label"]
                ] = match_accents[0]

        if match_words.any():
            assigned_df.at[row.Index, ["word_end", "word", "word_start"]] = match_words[
                0
            ]

        if match_tones.any():
            assigned_df.at[
                row.Index, ["ip_end", "bound_tone", "ip_start"]
            ] = match_tones[0]

        assigned_df["duration_est"] = assigned_df["end"] - assigned_df["start_est"]
        assigned_df.round({"duration_est": 4})

    return assigned_df


def filter_labels(annotation_df, annotation_type):
    """
    This function filters dataframes containing word or phone labels, dropping rows containing non-vowel or extralinguistic labels, respectively.
    """

    if annotation_type == "phones":
        label_inv = SAMPA_VOWELS
        keep_if = True  # keep the rows where elements of SAMPA_VOWELS are found

    elif annotation_type == "words":
        label_inv = EXTRALING_SOUNDS
        keep_if = False  # drop the rows where elements of EXTRALING_SOUNDS are found

    else:
        raise ValueError("Annotation type '{}' not supported.".format(annotation_type))

    filtered_df = annotation_df.loc[(annotation_df["label"].isin(label_inv)) == keep_if]

    return filtered_df


def find_peak_cands(intensity_obj, threshold):
    """
    Determine candidates for syllable nuclei by finding peaks in the PointProcess object
    """

    # Below lines are a direct functional translation from the Praat script by Wempe & de Jong
    intensity_mx = praat.call(intensity_obj, "Down to Matrix")
    snd_intensity_mx = praat.call(intensity_mx, "To Sound (slice)", 1)

    pt_proc_obj = praat.call(
        snd_intensity_mx, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70"
    )

    num_peaks = praat.call(pt_proc_obj, "Get number of points")
    times_peaks = [
        praat.call(pt_proc_obj, "Get time from index", i + 1) for i in range(num_peaks)
    ]
    vals_peaks = [
        praat.call(snd_intensity_mx, "Get value at time", time, "Cubic")
        for time in times_peaks
    ]

    peaks = [
        (times_peaks[i], vals_peaks[i])
        for i in range(num_peaks)
        if vals_peaks[i] > threshold
    ]

    return peaks


def validate(intensity_obj, peak_cands):
    """ This function validates the n potential peaks (i.e. potential syllable nuclei) that were found by checking whether they are:
        - followed by a min. 2dB dip (first peak)
        - surrounded by min. 2dB dip (second to penultimate peak) -- DISABLED
        - min. dB dip on any side (second to penultimate peak)
        - preceded by a min. 2dB dip (last peak)
    """

    valid_peaks = []

    for i in range(len(peak_cands) - 1):
        peak = peak_cands[i]

        if i == 0:
            next_peak = peak_cands[i + 1]
            next_intensity_dip = praat.call(
                intensity_obj, "Get minimum", peak[0], next_peak[0], "None"
            )
            intensity_diff = abs(peak[1] - next_intensity_dip)

            if intensity_diff > MIN_DIP_BETW_PEAKS:
                valid_peaks.append(peak)

        elif 0 < i < len(peak_cands) - 1:
            next_peak = peak_cands[i + 1]
            next_intensity_dip = praat.call(
                intensity_obj, "Get minimum", peak[0], next_peak[0], "None"
            )
            intensity_diff = abs(peak[1] - next_intensity_dip)

            if intensity_diff > MIN_DIP_BETW_PEAKS:
                # DISABLED: Possibility to require a dip before the nucleus as well.
                """prev_peak = peak_cands[i - 1]
                prev_intensity_dip = praat.call(
                    intensity_obj, "Get minimum", prev_peak[0], peak[0], "None"
                )
                intensity_diff = abs(peak[1] - prev_intensity_dip)

                if intensity_diff > MIN_DIP_BETW_PEAKS:"""
                valid_peaks.append(peak)
            else:
                prev_peak = peak_cands[i - 1]
                prev_intensity_dip = praat.call(
                    intensity_obj, "Get minimum", prev_peak[0], peak[0], "None"
                )
                intensity_diff = abs(peak[1] - prev_intensity_dip)

                if intensity_diff > MIN_DIP_BETW_PEAKS:
                    valid_peaks.append(peak)
        else:
            prev_peak = peak_cands[i - 1]
            prev_intensity_dip = praat.call(
                intensity_obj, "Get minimum", prev_peak[0], peak[0], "None"
            )
            intensity_diff = abs(peak[1] - prev_intensity_dip)

            if intensity_diff > MIN_DIP_BETW_PEAKS:
                valid_peaks.append(peak)

    return valid_peaks
