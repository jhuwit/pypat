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

`task="auto"` (the default) uses a binary model for an outcome with exactly two
distinct values; otherwise it fits a continuous regression model. Set
`task="binary"` or `task="continuous"` to choose explicitly.

The default PAT-L weights download once to `.pypat_weights/` in the current
directory. Pass `weights_path="/path/to/weights.h5"` to reuse an existing file
or choose the download destination. `X` must be a finite 2-D array with one
participant per row; the module pads its time axis to the model patch size,
scales using the training split only, and creates train/validation/test splits.

Install from a checkout with `pip install .` (or `pip install -e .` while
developing); this installs NumPy, scikit-learn, and TensorFlow. Once pushed to
GitHub, users can install directly with:

```bash
pip install "pypat @ git+https://github.com/OWNER/pypat.git"
```

After publishing a release to PyPI, installation is simply `pip install pypat`.
All runtime dependencies are declared in `pyproject.toml`.
