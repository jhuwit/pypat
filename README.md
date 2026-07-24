# pypat

Fine-tune the pretrained PAT accelerometry encoder with one function:

```python
import numpy as np
from pypat import fine_tune_pat

# X: (participants, minute-level measurements); y: one outcome per participant
result = fine_tune_pat(X, y, epochs=20, batch_size=32)
print(result.task, result.metrics)
new_predictions = result.predict(new_X)
```

To inspect the transformer's attention heads for a small set of participants:

```python
predictions, layer_attention = result.attention(X[:2])
# layer_attention[layer] has shape (participants, heads, patches, patches)
profile = result.attention_profile(X[:2], layer=-1)
# profile has one normalized score per original time point; plot profile[0]
```

## NHANES accelerometry data

Prepare complete seven-day NHANES records with:

```bash
PYTHONPATH=src python scripts/prepare_nhanes_ac.py
```

The script downloads PhysioNet's source CSV only when it is missing, then
creates an `X` matrix with one 10,080-minute row per participant and matching
`participant_ids`. Join your outcome data to those IDs to create `y`, then run
`fine_tune_pat(X, y)`.

The default sequence is days 2--8, so day 2 is its first day. Use
`--start-day 8` to make day 8 the first day (cyclically ordered as 8, 2, ..., 7).
For a non-cyclic custom ordering, use `--day-order 8,7,2,3,4,5,6`; this makes
day 8 the first and day 2 the third input day.

To rotate an already loaded `X` before sending it to the embedder/model:

```python
from pypat import rotate_nhanes_weekly_accelerometry

X_shifted = rotate_nhanes_weekly_accelerometry(X, start_day=8)
result = fine_tune_pat(X_shifted, y)
```

To make fine-tuning less sensitive to the first-day position, train on all
seven cyclic day rotations while retaining independent validation/test sets:

```python
result = fine_tune_pat(X, y, all_day_cycles=True)
```

This option requires seven-day (10,080-minute) records and augments only the
training split.

`task="auto"` (the default) uses a binary model for an outcome with exactly two
distinct values; otherwise it fits a continuous regression model. Set
`task="binary"` or `task="continuous"` to choose explicitly.

For a categorical outcome with three or more levels, specify a softmax head:

```python
result = fine_tune_pat(X, y_category, task="categorical")
labels = result.predict_classes(new_X)
```

For a time-to-event outcome, use discrete time bins and a censoring-aware
hazard head. Supply `y_survival` with two columns: zero-based `time_bin` and
`event_observed` (1 for an event, 0 for right-censoring):

```python
y_survival = np.column_stack([time_bin, event_observed])
result = fine_tune_pat(X, y_survival, task="survival", num_time_bins=12)
hazards = result.predict(new_X)  # one event hazard per time bin
```

The default PAT-L weights download once to your operating system's user cache.
Pass `weights_path="/path/to/weights.h5"` to reuse an existing file or choose
the download destination. `X` must be a finite 2-D array with one
participant per row; the module pads its time axis to the model patch size,
scales using the training split only, and creates train/validation/test splits.

Install from a checkout with `pip install .` (or `pip install -e .` while
developing); this installs NumPy, scikit-learn, and TensorFlow. Once pushed to
GitHub, users can install directly with:

```bash
pip install "pypat @ git+https://github.com/jhuwit/pypat.git"
```

After publishing a release to PyPI, installation is simply `pip install pypat`.
All runtime dependencies are declared in `pyproject.toml`.

## GPU covariate experiment

Install RDS support with `pip install -e ".[rds]"`, then run:

```bash
python scripts/finetune_gender_gpu.py --outcome-column gender --require-gpu --epochs 30 --batch-size 16
```

The script joins any selected RDS covariate column to complete activity weeks by `SEQN`, uses a
participant-level train/validation/test split, and saves ROC AUC, average
precision, accuracy, balanced accuracy, F1, log loss, a confusion matrix, test
predictions, and the fine-tuned weights under `runs/gender_gpu/`.

For a Colab GPU smoke test, use a small participant cohort and conservative
batch size; the full 10,080-minute attention model is memory intensive:

```bash
python scripts/finetune_gender_gpu.py \
  --max-participants 100 --epochs 5 --batch-size 2 --output-dir runs/gender_colab
```

When `--max-participants` is set, loading stops after a small buffer of complete
activity weeks is available, then selects the requested stratified cohort.

By default, a complete week has one record for every day 2--8. Both preparation
scripts report counts of complete, one-day-missing, more-incomplete, and
duplicate-day participants with `-v`. Add `--pad-one-missing-day` to retain
participants missing exactly one daily record; its entire 1,440-minute block is
filled with zeros.

Use `--task categorical` for integer-coded 3+ level outcomes (for example,
`race_hispanic_origin`). String outcomes with 3+ levels are categorical by
default; numeric outcomes with more than two values are continuous by default.

On JHPCE, submit [finetune_gender_jhpce.sbatch](scripts/finetune_gender_jhpce.sbatch)
from the project root after creating `.venv` as described at the top of that
file:

```bash
sbatch scripts/finetune_gender_jhpce.sbatch
```
