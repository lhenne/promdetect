# PromDetect

Neural network models for automatic pitch accent detection.

This repository accompanies my MA thesis in Linguistics, *Automatic Detection of Prosodic Prominence in Speech Using Neural Networks*.
It contains Python scripts for the models for syllable nucleus- and word-based prominence detection presented in the thesis.

Training data is derived from the DIRNDL corpus of German radio speech.
Only the immediate input data sets were uploaded due to licensing issues.

## Data preparation

Feature extraction and pre-processing scripts can be found in:

* The `prep` directory for nucleus-based training data,
* `frame_based` for frame-level training data,
* `word_based` for word-level training data

## Final models

The trained models can be found in the `models/model_store` directory.
Scripts for model construction, training and evaluation are located in separate subdirectories of `models`.
