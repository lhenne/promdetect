import pandas as pd
import parselmouth as pm
from parselmouth import praat


class FrameLevelExtractor:
    def __init__(self, wav_file, gender="f") -> None:
        self.wav_file = wav_file
        self.snd_obj = pm.Sound(self.wav_file)
        self.gender = gender
        self.features = pd.DataFrame()
        self.TIME_STEP = 0.01

        if gender == "f":
            self.__pitch_range = (75, 500)
        else:
            self.__pitch_range = (50, 300)

        self.pitch_obj = self.snd_obj.to_pitch_cc(
            time_step=self.TIME_STEP,
            pitch_floor=self.__pitch_range[0],
            pitch_ceiling=self.__pitch_range[1],
        )

        self.pitch_extraction()

    def pitch_extraction(self):
        timestamps = self.pitch_obj.ts()

        pitch_cands = self.pitch_obj.to_array().T
        best_cands = [max(frame, key=lambda x: x[1]) for frame in pitch_cands]
        f0, voicing_pr = zip(*best_cands)  # candidate strength as replacement for voicing probability, will be normalized later so high values are not problematic

        cols_to_add = ["time", "f0", "voicing_pr"]
        vals_to_add = zip(timestamps, f0, voicing_pr)

        data_to_add = pd.DataFrame(vals_to_add, columns=cols_to_add)
        self.features = pd.concat([self.features, data_to_add])

    def rms_extraction(self):

        self.features["rms"] = [
            self.snd_obj.get_rms(
                from_time=frame.time, to_time=(frame.time + self.TIME_STEP)
            )
            for frame in self.features.itertuples()
        ]

    def loudness_extraction(self):

        self.cochleagram = praat.call(
            self.snd_obj, "To Cochleagram", 0.01, 0.1, 0.03, 0.03
        )
        self.features["excitation"] = [
            praat.call(self.cochleagram, "To Excitation (slice)", frame.time)
            for frame in self.features.itertuples()
        ]

        self.features["loudness"] = [
            praat.call(frame.excitation, "Get loudness")
            for frame in self.features.itertuples()
        ]

        self.features.drop(columns="excitation")

    def zcr_extraction(self):

        self.features["snd_part"] = [
            self.snd_obj.extract_part(from_time=frame.time, to_time=frame.time + 0.01)
            for frame in self.features.itertuples()
        ]

        self.features["pp_part"] = [
            praat.call(frame.snd_part, "To PointProcess (zeroes)", 1, "yes", "yes")
            for frame in self.features.itertuples()
        ]

        self.features["zcr"] = [
            praat.call(frame.pp_part, "Get number of points")
            for frame in self.features.itertuples()
        ]

        self.features.drop(columns=["snd_part", "pp_part"])
