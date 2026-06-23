# Trustworthy Suicide Risk Detection

NLP pipeline for detecting suicide risk in social media text, built around
trustworthiness (calibration, conformal prediction and explainability) rather
than accuracy alone. Developed in phases, with documentation written alongside
the code.

## Status

- [x] **Preprocessing** — data loading, four way split, cleaning, TF-IDF, handcrafted features, slang normalisation
- [ ] Classical models (Logistic Regression, Naive Bayes, Linear SVM, Random Forest, XGBoost)
- [ ] Ensembles (weighted voting, stacking)
- [ ] Trustworthiness (calibration, RAPS conformal prediction)
- [ ] Explainability (SHAP, LIME, cross method agreement)
- [ ] Deep learning (BiLSTM, transformers)
- [ ] Proposed architecture (attention gated fusion)

## Project structure

```
config.py                  paths and constants, single source of truth
data/
  raw/                     place Suicide_Detection.csv here
  splits/                  generated train, val, calibration, test
  load_data.py             loading, filtering, four way split
preprocessing/
  slang_normaliser.py      algospeak and informal term normalisation
  text_cleaning.py         cleaning and TF-IDF
  feature_engineering.py   nineteen handcrafted features
scripts/
  run_preprocessing.py     runs the whole preprocessing end to end
outputs/                   generated cleaned text, features, vectoriser, stats
docs/                      documentation written alongside the code
```

## Preprocessing

The first stage takes the raw corpus to model ready features.

1. Load the raw CSV, filter it, build a four way stratified split.
2. Clean the text (lowercase, strip HTML and links, remove non letters, tokenise, remove stopwords, lemmatise).
3. Fit TF-IDF on the training set only, then transform the other splits.
4. Build nineteen handcrafted psycholinguistic features, plus a sixteen feature version for Naive Bayes.

A slang and algospeak normalisation step runs first, so coded crisis terms such
as `unalive`, `kms` and `sewer slide` are mapped to plain wording and become
visible to the model. It can be switched off in `config.py` to reproduce the
original pipeline and compare the two. Full write up in
[`docs/Data_and_Preprocessing.docx`](docs/Data_and_Preprocessing.docx).

### Run it

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
python scripts/run_preprocessing.py
```

Place `Suicide_Detection.csv` in `data/raw/` first. Outputs are written to
`outputs/`.

## Dataset

Public Suicide and Depression Detection dataset (Reddit, via Pushshift). After
filtering, 231,113 posts, close to balanced between the two classes. Raw data and
generated outputs are not tracked in Git, since they are large and reproducible.

## Notes

Built and run on Python 3.10. The TF-IDF vectoriser is fit on the training set
only to avoid leakage, and the calibration split is kept aside for conformal
prediction so its guarantee stays valid.
