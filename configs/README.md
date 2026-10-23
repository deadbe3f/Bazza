# Experiment settings

The default editable Python settings are in `../config.py`. Put alternative
Python settings here, exporting a `config = Config(...)` object. These files
contain values, paths and ordinary Python expressions; model construction stays
in train.py. Paths resolve relative to the selected settings file.

Use `--config /absolute/path/to/settings.py`. JSON is generated for durable run
records and optional Hugging Face export, not used as the training input.
