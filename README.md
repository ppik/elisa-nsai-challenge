# Solution for Elisa AI Challenge 2018

For more information about the challenge and data see
https://www.elisa.ee/et/elisa-ai-challenge and
https://gitlab.com/TeamMindTitan/workshop-naai

## Dependencies

Requires Python 3.6

This project uses [Pipfile](https://github.com/pypa/pipfile) to manage its
Python dependencies.

## Model Training

The model training can be performed using script `train_model.py`.
Training data file paths or urls (`normals.parquet` and 'probs.parquet') must be
provided as command line arguments.
Optionally these can be provided as an environment variable TRAIN_FILES as comma
separated list.
The resulting model will be saved under models directory together with training
log.

```
$ pipenv run python scripts/train_model.py data_file [data_file]...
```

## Inference

The inference script is `run_model.py`.

Both the model file path and data file path must be provided as command line
arguments.

If the model file is omitted, the latest model file from models directory
will be chosen.
Optionally the data file can be provided as an environment variable RUN_FILES
as comma separated list.

```
$ pipenv run python scripts/run_model.py [model_file] data_file
```

The results will be saved under models directory combining under as csv file
`basename(model_file)-basename(data_file).csv`
