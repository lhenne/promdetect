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
        input_file = "/home/lukas/Dokumente/Uni/ma_thesis/promdetect/tests/test_material/test.accents"

        correct_output = "    17.748754  121 H*L\n    19.194136  121 H*\n\n"

        self.assertEqual(process_annotations.read_file(input_file), correct_output)

    def test_clean_text(self):
        input_text = '    das\n      k"onnte besser auss"ahen\n\n'
        annotation_type = "words"

        correct_output = " das\n könnte besser aussähen\n\n"

        self.assertEqual(
            process_annotations.clean_text(input_text, annotation_type), correct_output
        )
