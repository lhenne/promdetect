"""
Code to test functions in the `prep` submodule
"""

import unittest
from promdetect.prep import process_annotations


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
        self.assertFalse("time" in output_df.columns)
        self.assertFalse("xwaves" in output_df.columns)
        self.assertTrue("label" in output_df.columns)
        self.assertTrue(len(output_df.index) == 3)

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
