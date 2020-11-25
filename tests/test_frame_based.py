import unittest
import numpy as np
from promdetect.frame_based import extract_frame_features


class FrameBasedExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wav_file = "/home/lukas/Dokumente/Uni/ma_thesis/quelldaten/DIRNDL-prosody/dlf-nachrichten-200703250000.wav"
        cls.tester = extract_frame_features.FrameLevelExtractor(cls.wav_file, "m")

    def test_pitch_extraction(cls):
        expected_f0_vals = np.around(
            [
                137.02502588552142,
                137.34105374768982,
                137.97088884779652,
                138.16917804734155,
                137.98379432570397,
                138.29299996864762,
            ],
            decimals=1,
        )
        true_f0_vals = np.around(
            cls.tester.features.iloc[26780:26786]["f0"], decimals=1
        )

        cls.assertTrue(np.array_equal(expected_f0_vals, true_f0_vals))

    def test_rms_extraction(cls):
        cls.tester.rms_extraction()

        expected_rms_vals = np.around(
            [
                0.22859686615956204,
                0.202701214094893,
                0.18102478728338137,
                0.22366511928857805,
                0.18798011519880028,
                0.20556932312396975,
            ],
            decimals=1,
        )
        true_rms_vals = np.around(
            cls.tester.features.iloc[26779:26785]["rms"], decimals=1
        )

        cls.assertTrue(np.array_equal(expected_rms_vals, true_rms_vals))

    def test_loudness_extraction(cls):
        cls.tester.loudness_extraction()

        expected_loudness_vals = np.around(
            [
                28.808457606109116,
                31.22651235885872,
                33.32895400798863,
                34.85523074885036,
                34.85523074885036,
                36.63164460593617,
                36.75379913302901,
            ],
            decimals=4,
        )
        true_loudness_vals = np.around(
            cls.tester.features.iloc[26779:26786]["loudness"], decimals=4
        )

        cls.assertTrue(np.array_equal(expected_loudness_vals, true_loudness_vals))

    def test_zero_crossing_rate_extraction(cls):
        cls.tester.zcr_extraction()

        expected_zcr_vals = np.array([5, 6, 6, 5, 8, 7])
        true_zcr_vals = np.array(cls.tester.features.iloc[26779:26785]["zcr"])

        cls.assertTrue(np.array_equal(expected_zcr_vals, true_zcr_vals))
