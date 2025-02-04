"""
Code to test functions in the `prep` submodule
"""

import unittest
from pandas import DataFrame
import numpy as np
import json
from parselmouth import Sound
from parselmouth import praat
from promdetect.prep import (
    process_annotations,
    find_syllable_nuclei,
    extract_features,
    prepare_data,
)


class AnnotationImportTests(unittest.TestCase):
    """
    Test that all imports from the annotation files work correctly.
    """

    def test_read_file(self):
        """
        Test read_file(): Are files imported with the correct formatting?
        """

        input_file = "tests/test_material/test.accents"

        correct_output = "    17.748754  121 H*L\n    19.194136  121 H*\n\n"

        self.assertEqual(process_annotations.read_file(input_file), correct_output)

    def test_clean_text(self):
        """
        Test clean_text(): Are strings reformatted and normalized correctly?
        """

        input_text = '    das\n      k"onnte besser auss"ahen\n\n'
        annotation_type = "words"

        correct_output = "das\nkönnte besser aussähen\n\n"

        self.assertEqual(
            process_annotations.clean_text(input_text, annotation_type), correct_output
        )

    def test_annotation_input_type(self):
        """
        Test get_annotation_data(): Do invalid input types produce an error?
        """

        annotation_file = "tests/test_material/test.phrases"

        with self.assertRaises(ValueError):
            tester = process_annotations.AnnotationReader(annotation_file)
            tester.get_annotation_data()

    def test_annotation_missing_file(self):
        """
        Test get_annotation_data(): Does an invalid file path produce an error?
        """

        annotation_file = "does/not/exist.accents"

        with self.assertRaises(FileNotFoundError):
            tester = process_annotations.AnnotationReader(annotation_file)
            tester.get_annotation_data()

    def test_annotation_accents_output(self):
        """
        Test get_annotation_data(): Does the function return the correct output for a file containing accent labels?
        """

        annotation_file = "tests/test_material/test.accents"

        tester = process_annotations.AnnotationReader(annotation_file)
        output_df = tester.get_annotation_data()

        self.assertTrue("time" in output_df.columns)
        self.assertFalse("end" in output_df.columns)
        self.assertFalse("xwaves" in output_df.columns)
        self.assertTrue("label" in output_df.columns)
        self.assertTrue(len(output_df.index) == 2)

    def test_annotation_words_output(self):
        """
        Test get_annotation_data(): Does the function return the correct output for a file containing word labels?
        """

        annotation_file = "tests/test_material/test.words"

        tester = process_annotations.AnnotationReader(annotation_file)
        output_df = tester.get_annotation_data()

        self.assertTrue("end" in output_df.columns)
        self.assertTrue("start_est" in output_df.columns)
        self.assertFalse("time" in output_df.columns)
        self.assertFalse("xwaves" in output_df.columns)
        self.assertTrue("label" in output_df.columns)
        self.assertTrue(len(output_df.index) == 3)

    def test_annotation_phones_output(self):
        """
        Test get_annotation_data(): Does the function return the correct output for a file containing phone labels?
        """

        annotation_file = "tests/test_material/test.phones"

        tester = process_annotations.AnnotationReader(annotation_file)
        output_df = tester.get_annotation_data()

        self.assertTrue("end" in output_df.columns)
        self.assertTrue("start_est" in output_df.columns)
        self.assertFalse("time" in output_df.columns)
        self.assertFalse("xwaves" in output_df.columns)
        self.assertTrue("label" in output_df.columns)
        self.assertTrue(len(output_df.index) == 5)

    def test_invalid_id(self):
        """
        Test get_speaker_info(): Does an invalid recording ID as input return the correct tuple?
        """

        recording_id = "notarealid2005-08-20-1500"

        expected_output = ("unknown", "f")

        tester = process_annotations.AnnotationReader(recording_id)
        output = tester.get_speaker_info()

        self.assertTrue(expected_output == output)

    def test_speaker_info(self):
        """
        Test get_speaker_info(): Does the function return the correct data for a given recording?
        """

        recording_id = "dlf-nachrichten-200703260600"
        correct_output = ("2", "f")

        tester = process_annotations.AnnotationReader(recording_id)

        self.assertTrue(tester.get_speaker_info() == correct_output)


class NucleiExtractionTests(unittest.TestCase):
    """
    Test that the automatic extraction of syllable nuclei works as expected.
    """

    def test_filter_vowel_sounds(self):
        """
        Are non-vowel labels filtered out correctly by the filter_labels function?
        """

        phones_df = DataFrame(
            [
                (26.130000, "h", 25.8701),
                (26.160000, "E", 26.1301),
                (26.270000, "_6", 26.1601),
                (26.360000, "k", 26.2701),
                (26.420000, "l", 26.3601),
                (26.490000, "E:", 26.4201),
                (26.540000, "_6", 26.4901),
                (26.620000, "t", 26.5401),
                (26.740000, "@", 26.6201),
            ],
            columns=["end", "label", "start_est"],
        )
        annotation_type = "phones"

        phones_df_filtered = find_syllable_nuclei.filter_labels(
            phones_df, annotation_type
        )

        self.assertTrue(len(phones_df_filtered) == 5)
        self.assertTrue(
            list(phones_df_filtered["label"]) == ["E", "_6", "E:", "_6", "@"]
        )

    def test_filter_breathing_sounds(self):
        """
        Are extralinguistic labels (breathing sounds) filtered out correctly by the filter_labels function?
        """

        words_df = DataFrame(
            [
                (25.870000, "bewirken", 25.3501),
                (26.130000, "[h]", 25.8701),
                (26.740000, "erklärte", 26.1301),
                (27.460000, "Außenminister", 26.7401),
            ],
            columns=["end", "label", "start_est"],
        )
        annotation_type = "words"

        words_df_filtered = find_syllable_nuclei.filter_labels(
            words_df, annotation_type
        )

        self.assertTrue(len(words_df_filtered) == 3)
        self.assertTrue(
            list(words_df_filtered["label"])
            == ["bewirken", "erklärte", "Außenminister"]
        )

    def test_point_label_assignment(self):
        """
        Are the nucleus points detected by the algorithm assigned to the correct word and phone labels?
        """

        nucleus_points = [26.15, 26.29, 26.46, 26.6201]

        phones_df = DataFrame(
            [
                (26.130000, "h", 25.8701),
                (26.160000, "E", 26.1301),
                (26.270000, "_6", 26.1601),
                (26.360000, "k", 26.2701),
                (26.420000, "l", 26.3601),
                (26.490000, "E:", 26.4201),
                (26.540000, "_6", 26.4901),
                (26.620000, "t", 26.5401),
                (26.740000, "@", 26.6201),
            ],
            columns=["end", "label", "start_est"],
        )

        words_df = DataFrame(
            [
                (25.870000, "bewirken", 25.3501),
                (26.130000, "[h]", 25.8701),
                (26.740000, "erklärte", 26.1301),
                (27.460000, "Außenminister", 26.7401),
            ],
            columns=["end", "label", "start_est"],
        )

        tones_df = DataFrame(
            [(25.87, "<P>", 22.0), (28.5, "%", 25.8701)],
            columns=["time", "label", "start_est"],
        )

        accents_df = DataFrame(
            [
                (26.515000, "LH*L"),
                (27.623440, "L*H"),
                (27.995732, "H*L"),
                (28.682547, "!H*L"),
            ],
            columns=["time", "label"],
        )

        assigned_df = find_syllable_nuclei.assign_points_labels(
            nucleus_points,
            phones=phones_df,
            words=words_df,
            tones=tones_df,
            accents=accents_df,
        )

        self.assertTrue(len(assigned_df) == 4)
        self.assertTrue(list(assigned_df["phone"]) == ["E", np.nan, "E:", "@"])
        self.assertTrue(
            list(assigned_df["word"])
            == ["erklärte", "erklärte", "erklärte", "erklärte"]
        )

    def test_timestamp_assignment(self):
        """
        Are the nucleus points detected by the algorithm assigned with the correct timestamps?
        Duration is not tested because for some reason identity testing for it is impossible.
        """

        nucleus_points = [26.15, 26.29, 26.46, 26.6201]

        phones_df = DataFrame(
            [
                (26.130000, "h", 25.8701),
                (26.160000, "E", 26.1301),
                (26.270000, "_6", 26.1601),
                (26.360000, "k", 26.2701),
                (26.420000, "l", 26.3601),
                (26.490000, "E:", 26.4201),
                (26.540000, "_6", 26.4901),
                (26.620000, "t", 26.5401),
                (26.740000, "@", 26.6201),
            ],
            columns=["end", "label", "start_est"],
        )

        words_df = DataFrame(
            [
                (25.870000, "bewirken", 25.3501),
                (26.130000, "[h]", 25.8701),
                (26.740000, "erklärte", 26.1301),
                (27.460000, "Außenminister", 26.7401),
            ],
            columns=["end", "label", "start_est"],
        )

        tones_df = DataFrame(
            [(25.87, "<P>", 22.0), (28.5, "%", 25.8701)],
            columns=["time", "label", "start_est"],
        )

        accents_df = DataFrame(
            [
                (26.515000, "LH*L"),
                (27.623440, "L*H"),
                (27.995732, "H*L"),
                (28.682547, "!H*L"),
            ],
            columns=["time", "label"],
        )

        assigned_df = find_syllable_nuclei.assign_points_labels(
            nucleus_points,
            phones=phones_df,
            words=words_df,
            tones=tones_df,
            accents=accents_df,
        )

        self.assertTrue(list(assigned_df["end"]) == [26.16, np.nan, 26.49, 26.74])
        self.assertTrue(
            list(assigned_df["start_est"]) == [26.1301, np.nan, 26.4201, 26.6201]
        )

    def test_nucleus_extraction(self):
        """
        Does get_nucleus_points find more than half of the syllable nuclei that were determined manually in the test file?
        """

        nuc_timestamps = [
            (0.4605, 0.5101),
            (0.7298, 0.7945),
            (0.8556, 0.9681),
            (1.2377, 1.3014),
            (1.3472, 1.4430),
            (1.5404, 1.5947),
            (1.7502, 1.8069),
        ]
        input_file = "tests/test_material/test.wav"

        nuc_points = find_syllable_nuclei.get_nucleus_points(input_file)

        matches = [
            point
            for point in nuc_points
            for span in nuc_timestamps
            if (span[0] <= point <= span[1])
        ]

        self.assertTrue(len(matches) >= len(nuc_timestamps) / 2)

    def test_find_peak_candidates(self):
        """
        Does find_peak_cands find all the peaks it can be expected to find?
        """

        snd_obj = Sound("tests/test_material/test.wav")
        snd_denoised = praat.call(
            snd_obj, "Remove noise", 0, 0, 0.025, 75, 10_000, 40, "Spectral subtraction"
        )
        intensity_obj = snd_denoised.to_intensity(minimum_pitch=75)

        threshold = 60.13517359983467

        peak_candidates = find_syllable_nuclei.find_peak_cands(intensity_obj, threshold)

        self.assertTrue(len(peak_candidates) == 16)
        self.assertAlmostEqual(peak_candidates[5][1], 70.635, places=3)


class FeatureExtractionTests(unittest.TestCase):
    """
    Do the various feature extraction functions return the results they are expected to return?
    """

    def test_rms_extraction(self):
        """
        Does the RMS calculation using Praat work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df)

        rms = np.around(tester.get_rms(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    0.17689167172400735,
                    0.2278562589839203,
                    0.16188228973962826,
                    0.1256124165026432,
                    0.1446231088144042,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(rms, expected_vals, equal_nan=True))

    def test_intensity_min_extraction(self):
        """
        Is the mininum intensity extracted correctly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_intensity()

        intensities = np.around(tester.get_min_intensity_nuclei(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    72.52231907753077,
                    78.54116284340166,
                    75.2261318921721,
                    70.07862900229256,
                    76.88232010504962,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(intensities, expected_vals, equal_nan=True))

    def test_intensity_min_pos_extraction(self):
        """
        Is the relative position of the intensity minimum in the syllable nucleus extracted correctly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_intensity()

        pos = np.around(tester.get_min_intensity_pos(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    0.14431818181812883,
                    0.043055555555504624,
                    0.0645833333332859,
                    0.1410714285713439,
                    0.42343749999993424,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(pos, expected_vals, equal_nan=True))

    def test_intensity_max_pos_extraction(self):
        """
        Is the relative position of the intensity maximum in the syllable nucleus extracted correctly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_intensity()

        pos = np.around(tester.get_max_intensity_pos(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    0.8715909090908767,
                    0.39861111111106107,
                    0.5979166666666152,
                    0.5982142857142112,
                    0.023437499999934254,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(pos, expected_vals, equal_nan=True))

    def test_intensity_std_extraction(self):
        """
        Is the standard deviation for intensity extracted correctly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_intensity()

        stds = np.around(tester.get_intensity_std_nuclei(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    3.369907965314268,
                    1.1402104298852436,
                    1.5638238156409505,
                    3.419440119923512,
                    0.5701671691859589,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(stds, expected_vals, equal_nan=True))

    def test_local_pitch_slope(self):
        """
        Is the absolute slope (without octave jumps) calculated correctly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")

        pitch_slope = np.around(tester.get_pitch_slope(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    26.819895462113966,
                    10.194947721704583,
                    15.767233863040769,
                    21.886583599912406,
                    32.763328999228335,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(pitch_slope, expected_vals, equal_nan=True))

    def test_intensity_extraction_nuclei_max(self):
        """
        Does the intensity extraction (dB) using Praat work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df)
        tester.calc_intensity()

        intensities = np.around(tester.get_max_intensity_nuclei(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    81.1521732084371,
                    81.72280626772428,
                    79.37680512029912,
                    77.91419036596378,
                    78.403839787533,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(intensities, expected_vals, equal_nan=True))

    def test_intensity_extraction_ip_mean(self):
        """
        Does the intensity extraction (dB) for intonation phrases work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I", 11.41, 13.91),
                (13.8, 14.1, "e:", 13.911, 16.2),
                (16.4, 16.7, "I", 16.201, 18.91),
            ],
            columns=["start_est", "end", "phone", "ip_start", "ip_end"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df)
        tester.calc_intensity()

        intensities = np.around(tester.get_intensity_ip(), decimals=4)

        expected_vals = np.around(
            np.array([75.39665468978198, 71.55360893937386, 70.32496318148395]),
            decimals=4,
        )

        self.assertTrue(np.array_equal(intensities, expected_vals, equal_nan=True))

    def test_f0_extraction_nuclei_max(self):
        """
        Does the F0 peak extraction for syllable nuclei work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_pitch()

        f0s = np.around(tester.get_f0_max_nuclei(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    117.87253295571392,
                    123.1675694877848,
                    134.76548248292062,
                    136.5251634214678,
                    151.56666619140884,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(f0s, expected_vals, equal_nan=True))

    def test_f0_extraction_nuclei_min(self):
        """
        Does the F0 minimum extraction for syllable nuclei work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_pitch()

        f0s = np.around(tester.get_f0_min_nuclei(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    110.82714523349438,
                    113.2546976292171,
                    128.16975339751528,
                    124.73456144344753,
                    138.3529433156357,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(f0s, expected_vals, equal_nan=True))

    def test_f0_extraction_nuclei_range(self):
        """
        Does the F0 range extraction for syllable nuclei work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_pitch()
        tester.get_f0_min_nuclei()
        tester.get_f0_max_nuclei()

        f0s = np.around(tester.get_f0_range_nuclei(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    7.0453877222195445,
                    9.912871858567698,
                    6.595729085405338,
                    11.790601978020263,
                    13.21372287577313,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(f0s, expected_vals, equal_nan=True))

    def test_f0_extraction_nuclei_std(self):
        """
        Does the F0 standard deviation extraction for syllable nuclei work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_pitch()

        f0s = np.around(tester.get_f0_std_nuclei(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    2.280839592346914,
                    2.202496205316962,
                    2.411086034540446,
                    4.471311329355762,
                    5.209220274575025,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(f0s, expected_vals, equal_nan=True))

    def test_min_pitch_pos_nuclei(self):
        """
        Does the extraction of the relative position of the pitch minimum within syllable nuclei work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_pitch()

        pos = np.around(tester.get_f0_min_pos(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    0.6079545454545124,
                    0.020833333333274945,
                    0.031249999999941713,
                    0.9553571428571064,
                    0.27343749999995093,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(pos, expected_vals, equal_nan=True))

    def test_max_pitch_pos_nuclei(self):
        """
        Does the extraction of the relative position of the pitch minimum within syllable nuclei work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_pitch()

        pos = np.around(tester.get_f0_max_pos(), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    0.3352272727272441,
                    0.35416666666662144,
                    0.7812499999999343,
                    0.45535714285709367,
                    0.9609374999999467,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(pos, expected_vals, equal_nan=True))

    def test_pitch_excursion_ip(self):
        """
        Does the extraction of pitch excursion across the intonation phrase work correctly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I", 11.41, 13.91, 117.87253295571392),
                (11.63, 11.72, "e:", 11.41, 13.91, 123.1675694877848),
                (11.79, 11.85, "I", 11.41, 13.91, 134.76548248292062),
                (11.96, 12.03, "U", 11.41, 13.91, 136.5251634214678),
                (12.16, 12.24, "o:", 11.41, 13.91, 151.56666619140884),
            ],
            columns=["start_est", "end", "phone", "ip_start", "ip_end", "f0_max"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_pitch()

        excursions = np.around(tester.get_excursion("ip"), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    4.729683832131741,
                    5.4904221848570725,
                    7.04836523259897,
                    7.272955525472149,
                    9.082382906784996,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(excursions, expected_vals, equal_nan=True))

    def test_pitch_excursion_word(self):
        """
        Does the extraction of pitch excursion across the word work correctly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I", 11.41, 11.6, 117.87253295571392),
                (11.63, 11.72, "e:", 11.6, 11.75, 123.1675694877848),
                (11.79, 11.85, "I", 11.75, 12.3, 134.76548248292062),
                (11.96, 12.03, "U", 11.75, 12.3, 136.5251634214678),
                (12.16, 12.24, "o:", 11.75, 12.3, 151.56666619140884),
            ],
            columns=["start_est", "end", "phone", "word_start", "word_end", "f0_max"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")
        tester.calc_pitch()

        excursions = np.around(tester.get_excursion("word"), decimals=4)

        expected_vals = np.around(
            np.array(
                [
                    0.9820240572061552,
                    1.756133256029977,
                    3.216669606719651,
                    3.44125989959283,
                    5.2506872809056775,
                ]
            ),
            decimals=4,
        )

        self.assertTrue(np.array_equal(excursions, expected_vals, equal_nan=True))

    def test_normalized_duration_ip(self):
        """
        Are normalized durations relative to remainder of the intonation phrase extracted correctly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I", 11.41, 13.91),
                (11.63, 11.72, "e:", 11.41, 13.91),
                (11.79, 11.85, "I", 11.41, 13.91),
                (11.96, 12.03, "U", 11.41, 13.91),
                (12.16, 12.24, "o:", 11.41, 13.91),
            ],
            columns=["start_est", "end", "phone", "ip_start", "ip_end"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df)

        durations = np.around(tester.get_duration_normed(), decimals=4)

        expected_vals = np.around(
            np.array([1.3414634, 1.097561, 0.73170732, 0.85365854, 0.97560976]),
            decimals=4,
        )

        self.assertTrue(np.array_equal(durations, expected_vals, equal_nan=True))

    def test_spectral_tilt_mean(self):
        """
        Does the extraction of spectral tilt work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")

        tilt = np.around(
            tester.get_spectral_tilt_mean(), decimals=0
        )  # crude rounding factor due to problem with disparity between Praat GUI and API commands, not fatal

        expected_vals = np.around(
            np.array(
                [
                    142.71001735123184,
                    238.82379641896938,
                    209.10863223758525,
                    247.9976699000063,
                    299.64676965508664,
                ]
            ),
            decimals=0,  # crude rounding factor due to problem with disparity between Praat GUI and API commands, not fatal
        )

        self.assertTrue(np.array_equal(tilt, expected_vals, equal_nan=True))

    def test_spectral_tilt_range(self):
        """
        Does the extraction of spectral tilt range work properly?
        """

        wav_file = "tests/test_material/feature_extraction/test.wav"

        nuclei_df = DataFrame(
            [
                (11.41, 11.52, "I"),
                (11.63, 11.72, "e:"),
                (11.79, 11.85, "I"),
                (11.96, 12.03, "U"),
                (12.16, 12.24, "o:"),
            ],
            columns=["start_est", "end", "phone"],
        )

        tester = extract_features.Extractor(wav_file, nuclei=nuclei_df, gender="m")

        tilt = np.around(
            tester.get_spectral_tilt_range(), decimals=0
        )  # crude rounding factor due to problem with disparity between Praat GUI and API commands, not fatal

        expected_vals = np.around(
            np.array(
                [
                    149.61228824324905,
                    103.69618917064011,
                    76.79120165132159,
                    195.89299491267772,
                    125.97013425133713,
                ]
            ),
            decimals=0,  # crude rounding factor due to problem with disparity between Praat GUI and API commands, not fatal
        )

        self.assertTrue(np.array_equal(tilt, expected_vals, equal_nan=True))


class FeatureSetTests(unittest.TestCase):
    """
    Tests the functions in prepare_data.py
    """

    def test_run_method_from_config(self):
        """
        Can methods be called from the config successfully?
        """

        with open("tests/test_material/config_rms.json", "r") as cfg_file:
            cfg = json.load(cfg_file)

        tester = prepare_data.FeatureSet(cfg, "test")
        data = tester.run_config()

        self.assertTrue(isinstance(data, DataFrame))
        self.assertTrue("rms" in data.columns)

    def test_run_excursion_with_arguments(self):
        """
        Can the excursion method be run with arguments?
        """

        with open("tests/test_material/config_excursion_args.json", "r") as cfg_file:
            cfg = json.load(cfg_file)

        tester = prepare_data.FeatureSet(cfg, "test")
        data = tester.run_config()

        self.assertTrue(isinstance(data, DataFrame))
        self.assertTrue("excursion_word" in data.columns)
        self.assertTrue("excursion_ip" in data.columns)

    def test_run_multi_methods(self):
        """
        Can the class methods call multiple methods without any issues?
        """

        with open("tests/test_material/config_multi.json", "r") as cfg_file:
            cfg = json.load(cfg_file)

        tester = prepare_data.FeatureSet(cfg, "test")
        data = tester.run_config()

        self.assertTrue(isinstance(data, DataFrame))
        self.assertTrue("rms" in data.columns)
        self.assertTrue("duration_normed" in data.columns)
        self.assertTrue("max_intensity_nuclei" in data.columns)
