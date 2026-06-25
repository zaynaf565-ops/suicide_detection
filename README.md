# Trustworthy Suicide Risk Detection from Social Media

### Classical Machine Learning Phase

This project reads posts from social media and decides whether each post shows a risk of suicide or not. It is built to be careful as well as accurate. On top of the usual prediction it adds three safety features. It reports honest confidence levels, it can say "I am not sure about this one" and hand the post to a human, and it can show which words led to each decision. Everything here uses classical machine learning, which is fast and easy to inspect. Deep learning is a later phase and is not part of this repository yet.

This guide is written so that someone with no programming background can run the whole thing. Follow it from top to bottom and you will be fine.

---

## You can look at the results without running anything

The repository already includes the results from a full run. If you only want to see the outcome, open these folders and look at the images and the small spreadsheet files inside:

* `Classical_Models/results/plots` for how the five models compare
* `Trustworthiness/results/plots` for the confidence and abstention charts
* `Explainability/results/plots` for the word level explanations
* `Final_Evaluation/results` for the final scores on the untouched test set
* `EDA/results` for word clouds and overviews of the data
* `Slang_Comparison/results` and `stacking-and-ablation` for the extra studies

You only need the rest of this guide if you want to run the project yourself and produce these results from scratch.

---

## What you need before you start

1. A computer running Windows, macOS or Linux.
2. **Python 3.11**. You can get it from python.org. During installation on Windows, tick the box that says "Add Python to PATH".
3. **Git**, which is used to download the code. You can get it from git-scm.com. If you prefer not to install Git, you can download the code as a ZIP file from the green "Code" button on the GitHub page and skip the clone step.
4. **The dataset**. The data is not stored in this repository because it is large. Download the file named `Suicide_Detection.csv` from the public Kaggle dataset "Suicide and Depression Detection" by Nikhileswar Komati. It is about one hundred and sixty megabytes. You will be told where to put it in the setup below.

---

## How the project is organised

Each folder is one stage of the work. A stage reads what the previous stage produced and writes its own results. The table below is the map. The third column is the one file you run for that stage.

| Folder | What it does | File you run |
|---|---|---|
| `Pre_Processing` | cleans the posts, splits the data, builds the features | `Pre_Processing/scripts/run_preprocessing.py` |
| `Classical_Models` | trains the five models and tunes them | `Classical_Models/scripts/run_training.py` |
| `Ensembles` | combines the models into a voting and a stacking model | `Ensembles/run_ensemble.py` |
| `Trustworthiness` | confidence calibration and the abstain mechanism | `Trustworthiness/predict_test.py` then `Trustworthiness/run_trustworthiness.py` |
| `Final_Evaluation` | the final scores on the test set, with intervals | `Final_Evaluation/run_evaluation.py` |
| `EDA` | word clouds and data overviews | `EDA/run_eda.py` |
| `Explainability` | which words drove each decision | `Explainability/run_explainability.py` |
| `Slang_Comparison` | a heavy study, with and without slang cleaning | `Slang_Comparison/run_slang_off.py` |
| `stacking-and-ablation` | a heavy study, runs on Kaggle | `stacking-and-ablation/stacking-and-ablation-kaggle.ipynb` |

Inside most folders you will also see a `config.py`, which holds the settings, and a `results` folder, which is where the outputs land.

---

## Setup, which you do only once

### Step 1. Get the code onto your computer

Open a terminal. On Windows this is the app called Command Prompt or PowerShell. On macOS it is the app called Terminal. Then type the following and press Enter.

```bash
git clone https://github.com/zaynaf565-ops/suicide_detection.git
cd suicide_detection
```

If you downloaded the ZIP instead, unzip it, then in the terminal move into the unzipped folder using the `cd` command.

From here on, every command is run from inside this folder, which we call the project root.

### Step 2. Create a clean Python environment and install everything

This keeps the project tidy and separate from the rest of your computer. The second line installs every library the whole project needs, all from the single `requirements.txt` file in the project root.

```bash
python -m venv venv
```

Now turn the environment on. On Windows type:

```bash
venv\Scripts\activate
```

On macOS or Linux type:

```bash
source venv/bin/activate
```

Then install:

```bash
pip install -r requirements.txt
```

> If you ever see an error that mentions numpy 2, run `pip install "numpy<2"` and try again. Some library quietly upgraded numpy and this puts it back. This is the single most common hiccup, so keep it in mind.

### Step 3. Download the language data

The text cleaning needs a few small language files. Run this once.

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 4. Put the dataset in place

Take the `Suicide_Detection.csv` file you downloaded and place it here:

```
Pre_Processing/data/raw/Suicide_Detection.csv
```

The folder already exists in the repository, so you only need to drop the file into it.

You are now ready to run the project.

---

## Running the whole project, in order

Run these from the project root, one after another. Each command waits until it finishes, so let one complete before starting the next. The two heavy stages have their own sections below.

### Step 1. Preprocessing

```bash
python Pre_Processing/scripts/run_preprocessing.py
```

This cleans every post, splits the data into four parts, and builds the features. It writes everything to `Pre_Processing/outputs`. This takes a while because the dataset is large, so let it run. Everything after this depends on it.

### Step 2. Train the five models

```bash
python Classical_Models/scripts/run_training.py
```

This is the slowest single step, because it searches for the best settings of each model. On an ordinary laptop the search can take a few hours. It saves the trained models and the comparison plots inside `Classical_Models`.

### Step 3. Build the ensembles

```bash
python Ensembles/run_ensemble.py
```

This combines the five models into a voting model and a stacking model.

### Step 4. Confidence and the abstain mechanism

Run these two in order.

```bash
python Trustworthiness/predict_test.py
python Trustworthiness/run_trustworthiness.py
```

The first one scores the test set. The second one calibrates the confidence levels and builds the abstain sets, then saves the reliability and abstention charts.

### Step 5. Final evaluation

```bash
python Final_Evaluation/run_evaluation.py
```

This produces the final scores on the test set, with confidence intervals and a significance test.

### Step 6. Data overview charts

```bash
python EDA/run_eda.py
```

This makes the word clouds and the data overview heatmaps.

### Step 7. Explanations

```bash
python Explainability/run_explainability.py
```

This shows which words drove the decisions, and compares two different explanation methods. It shows a progress bar. If it feels slow you can lower the sample sizes near the top of the file, but the defaults are fine.

After these seven steps the main pipeline is complete and every results folder is filled.

---

## The first heavy study, slang comparison

This study cleans the data twice and trains the models twice, once with slang cleaning on and once with it off, so it runs for around two hours. It does not depend on the steps above and can be run whenever you have the time.

```bash
python Slang_Comparison/run_slang_off.py
```

The comparison table and chart land in `Slang_Comparison/results`.

---

## The second heavy study, stacking and ablation, on Kaggle

This study retrains models many times, so it is meant to run on Kaggle rather than on a laptop. You do not redo the preprocessing there. Instead you upload the feature files that preprocessing already produced.

First, take these two folders from your run:

```
Pre_Processing/outputs/tfidf
Pre_Processing/outputs/features
```

Upload them as a new Kaggle dataset, keeping both folders and their files exactly as they are. Then open the notebook `stacking-and-ablation/stacking-and-ablation-kaggle.ipynb` on Kaggle, attach that dataset to it, and choose Run all. The notebook finds the files on its own and produces the stacking and ablation charts, which you can download from the Kaggle output panel.

---

## Where the results appear

| Stage | Look here |
|---|---|
| Preprocessing | `Pre_Processing/outputs` |
| Classical models | `Classical_Models/results` |
| Ensembles | `Ensembles/results` |
| Confidence and abstain | `Trustworthiness/results` |
| Final evaluation | `Final_Evaluation/results` |
| Data overview | `EDA/results` |
| Explanations | `Explainability/results` |
| Slang study | `Slang_Comparison/results` |
| Stacking and ablation | the Kaggle output panel |

---

## If something goes wrong

**An error that mentions numpy 2.** Run `pip install "numpy<2"` and run your command again. This is the most common issue.

**An error that mentions punkt, stopwords or wordnet.** The language data did not download. Run the download line from Step 3 of the setup again.

**A message that it cannot find `Suicide_Detection.csv`.** The dataset is not in place. Check that the file sits exactly at `Pre_Processing/data/raw/Suicide_Detection.csv` and that the name matches.

**A message that begins with ModuleNotFound.** The environment is probably not turned on, or the libraries are not installed. Turn the environment on again with the activate command from Step 2, then run `pip install -r requirements.txt` once more.

**The training step runs out of memory or feels stuck.** The dataset is large, so the model search is genuinely slow on a small machine. Give it time, close other heavy programs, or run that stage on a machine with more memory such as a free Kaggle or Colab session.

---

## The whole thing, as a short checklist

1. Install Python 3.11 and Git.
2. `git clone` the repository and `cd` into it.
3. Create the environment, turn it on, and run `pip install -r requirements.txt`.
4. Run the language data download line.
5. Place `Suicide_Detection.csv` in `Pre_Processing/data/raw`.
6. Run the seven steps in order, from preprocessing to explanations.
7. When you have time, run the slang study, and run the Kaggle notebook for stacking and ablation.
8. If you ever see a numpy 2 error, run `pip install "numpy<2"`.

That is everything. Take it one step at a time and the project will run from start to finish.
