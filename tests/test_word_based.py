import unittest
import numpy as np
from promdetect.word_based import segmentation


class SegmentationTests(unittest.TestCase):
    def test_include_annotations(self):
        tester = segmentation.Segmenter(
            "words", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()

        assert "dlf-nachrichten-200703250000" in tester.annotations.keys()

    def test_output_type(self):
        tester = segmentation.Segmenter(
            "words", "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody"
        )

        tester.read_annotations()

        assert isinstance(tester.annotations, dict)
        assert isinstance(
            tester.annotations["dlf-nachrichten-200703250000"], np.ndarray
        )
