# Supply Chain - Customer Satisfaction

**Difficulty:** 6/10 &middot; **Task type:** Classification / NLP

This repo is one of eight DataScientest catalogue projects completed as DS practice deliverables during my Liora MLE Weiterbildung (Apr-Oct 2026).

## Repo layout

```
notebooks/      Jupyter notebooks (EDA, modelling, evaluation)
reports/        Markdown deliverables (exploration, modelling, architecture)
src/            Reusable Python modules
deliverables/   Final exports (HTML, PDF, slides)
manuscripts/    Long-form write-ups (where applicable)
brief.pdf       Original DataScientest project brief
```

## Data

Raw data is **not committed** (see `.gitignore`). The dataset for this project comes from the DataScientest catalogue. Place files locally under `data/` to reproduce.

## How to reproduce

```bash
git clone https://github.com/Sandyyy123/liora-02-supply-chain-csat.git
cd liora-02-supply-chain-csat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # if present
# place dataset under data/
jupyter lab notebooks/
```

## Status

Work in progress as of 2026-05. EDA + modelling notebooks present; reports tracked in `reports/`.

## License

MIT - see [LICENSE](LICENSE).

---

*Part of an 8-project portfolio. See companion repos: liora-01 through liora-08.*
