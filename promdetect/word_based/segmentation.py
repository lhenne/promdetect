from glob import glob
from pathlib import Path
from promdetect.prep.process_annotations import read_file, clean_text
from io import StringIO
import numpy as np
import pandas as pd


class Segmenter:
    def __init__(self, level, directory):
        self.level = level
        self.directory = directory
        self.files = glob(f"{directory}/*.{level}")

    def read_annotations(self):
        self.annotations = {}
        for file in self.files:
            recording = Path(file).stem
            raw_content = read_file(file)
            content = StringIO(clean_text(raw_content, self.level))
            self.annotations[recording] = self.content_to_np(content)

    def content_to_np(self, content):
        content_as_df = pd.read_csv(
            content,
            sep=" ",
            engine="python",
            quoting=3,
            names=["end", "xwaves", "label"],
        ).drop(  # Drop unnecessary xwaves column
            ["xwaves"], axis=1
        )

        content_as_df["start_est"] = np.nan
        num_rows = len(content_as_df.index)

        # Add data for estimated start timestamps of each word/int.phrase, which are 1ms after the end timestamp of the previous label (to avoid overlap for now). Set the starting time of the first label to N/A.
        for i in content_as_df.index:
            if i <= num_rows and i > 0:
                content_as_df.at[i, "start_est"] = (
                    content_as_df.loc[i - 1, "end"] + 0.0001
                )
            else:
                continue

        return content_as_df.to_numpy()

    def save_output(self, file):
        pass
