# Usage Instructions

## Step 1: Preprocess Data

Before training, you must preprocess the data:

```bash
python prepare_data.py
```

This will generate the processed dataset under `datas/processed`.

## Step 2: Train the Model

```bash
python train.py
```

Results will be saved in `datas/results`.

## Tips

- Modify constants like file paths in `const.py`
- Customize model in `model.py`