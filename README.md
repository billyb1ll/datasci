# Music Genre Classification (CRNN + MLP)

A small, practical repo for classifying music genres using two complementary approaches:

- CRNN on Mel-spectrograms (interactive in `notebooks/main2.ipynb`)
- MLP on precomputed tabular features (interactive in `notebooks/main3.ipynb`)

Both workflows save models/plots and include evaluation with a confusion matrix.

## Setup

- Python 3.10+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Optional (recommended for broader audio support):

```bash
# macOS
brew install ffmpeg
```

## Data

- Mel-spectrogram workflow expects GTZAN-style folders at:
  - `data/Data/genres_original/<genre>/*.wav`
- Features workflow expects a CSV:
  - `data/Data/features_30_sec.csv` (or `features_3_sec.csv`)

Adjust paths inside the notebooks if your layout differs.

## Notebooks

- `notebooks/main2.ipynb` (CRNN)
  - Extracts Mel-spectrograms with librosa
  - Builds a CRNN (Conv2D → BiLSTM) and trains with augmentation
  - Saves: `models/crnn_music_genre_model_best.keras`, `models/crnn_music_genre_model_final.keras`, training history, and confusion matrix

- `notebooks/main3.ipynb` (MLP baseline on CSV features)
  - Loads `features_30_sec.csv`
  - Splits into train/val/test, scales features, trains a small MLP
  - Saves: `models/features_scaler.joblib` and a model in `.keras` format (or exports a SavedModel if needed)

Open the notebooks in VS Code or Jupyter and run cells top to bottom.

## Outputs

- Models: `models/`
- Figures: `figures/` and/or repo root (confusion_matrix.png, training history)
- Logs: `logs/`

## Troubleshooting

- Saving model fails with `h5py` error:
  - Prefer saving with the `.keras` format (built-in). If that fails in your environment, `main3.ipynb` falls back to `model.export()` (Keras 3) or `tf.saved_model.save()` automatically.
  - To use HDF5 (`.h5`) explicitly, ensure `h5py` is installed.

- Audio decoding issues (librosa/audioread):
  - Install `ffmpeg` (see above).
  - The loader in `main2.ipynb` has fallbacks to pydub + resampy.

- GPU memory errors:
  - `main2.ipynb` enables TensorFlow memory growth if a GPU is present.
  - Reduce batch size if you encounter OOM.

## Optional .gitignore

To avoid committing large audio files:

```gitignore
audio/*.mp3
audio/*.wav
```

## Notes

- Default dataset path: `data/Data/genres_original`
- Default CSV path: `data/Data/features_30_sec.csv`
- You can tweak model sizes, augmentation strength, and training epochs in the notebooks.
