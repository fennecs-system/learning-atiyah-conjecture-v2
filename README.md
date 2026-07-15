# learning atiyah

Using PatternBoost (transformer generation + local search) to search for
extremal configurations of points in the plane relevant to Atiyah's
conjecture. See `main.tex` for the mathematical background (the "cell
classification" construction in `utils.compute_dots` is what's implemented
here).

## Setup

```
pip install -r requirements.txt
```

For running the tests too:

```
pip install -r requirements-dev.txt
```

## Usage

Generate a training set of random point configurations:

```
python gen_data.py --num-samples 100000 --out data.txt
```

Train the model, running the PatternBoost generate-then-local-search loop:

```
python makemore.py --input-file data.txt -b 64
```

Sample from (and resume) a trained model:

```
python makemore.py --work-dir out/run-<timestamp> --sample-only
```

## Tests

```
pytest
```
