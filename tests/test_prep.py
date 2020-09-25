"""
Code to test functions in the `prep` submodule
"""

from promdetect.prep.find_syllable_nuclei import find_peak_cands
import unittest
from pandas import DataFrame
import numpy as np
import parselmouth as pm
from parselmouth import praat
from promdetect.prep import process_annotations, find_syllable_nuclei


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
            process_annotations.get_annotation_data(annotation_file)

    def test_annotation_missing_file(self):
        """
        Test get_annotation_data(): Does an invalid file path produce an error?
        """

        annotation_file = "does/not/exist.accents"

        with self.assertRaises(FileNotFoundError):
            process_annotations.get_annotation_data(annotation_file)

    def test_annotation_accents_output(self):
        """
        Test get_annotation_data(): Does the function return the correct output for a file containing accent labels?
        """

        annotation_file = "tests/test_material/test.accents"

        output_df = process_annotations.get_annotation_data(annotation_file)

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

        output_df = process_annotations.get_annotation_data(annotation_file)

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

        output_df = process_annotations.get_annotation_data(annotation_file)

        self.assertTrue("end" in output_df.columns)
        self.assertTrue("start_est" in output_df.columns)
        self.assertFalse("time" in output_df.columns)
        self.assertFalse("xwaves" in output_df.columns)
        self.assertTrue("label" in output_df.columns)
        self.assertTrue(len(output_df.index) == 5)

    def test_invalid_id(self):
        """
        Test get_speaker_info(): Does an invalid recording ID as input raise the appropriate error?
        """

        recording_id = "notarealid2005-08-20-1500"

        with self.assertRaises(ValueError):
            process_annotations.get_speaker_info(recording_id)

    def test_speaker_info(self):
        """
        Test get_speaker_info(): Does the function return the correct data for a given recording?
        """

        recording_id = "200703260600"
        correct_output = ("2", "f")

        self.assertTrue(
            process_annotations.get_speaker_info(recording_id) == correct_output
        )


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

        assigned_df = find_syllable_nuclei.assign_points_labels(
            nucleus_points, phones=phones_df, words=words_df
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

        assigned_df = find_syllable_nuclei.assign_points_labels(
            nucleus_points, phones=phones_df, words=words_df
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
        snd_obj = pm.Sound("tests/test_material/test.wav")
        snd_denoised = praat.call(
            snd_obj, "Remove noise", 0, 0, 0.025, 75, 10_000, 40, "Spectral subtraction"
        )
        intensity_obj = snd_denoised.to_intensity(minimum_pitch=75)

        threshold = 60.13517359983467

        peak_candidates = find_peak_cands(intensity_obj, threshold)

        self.assertTrue(len(peak_candidates) == 16)
        self.assertAlmostEqual(peak_candidates[5][1], 70.635, places=3)
