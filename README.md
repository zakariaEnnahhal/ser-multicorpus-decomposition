# Decomposing Pooled Multi-Corpus Speech Emotion Recognition Performance

Code for the paper:

> Z. Ennahhal, M. Iaousse, M. El Kamili, I. Berrada.
> **Decomposing Pooled Multi-Corpus Speech Emotion Recognition Performance: Speaker Overlap, Corpus Effects and Run-to-Run Variation.**
> Submitted to *Intelligent Systems with Applications*, 2026.

The study trains a tri-branch speech emotion recognition (SER) model — frozen WavLM Base+ representations, a Transformer-based log-mel branch and 75 handcrafted acoustic–prosodic features, combined by gated fusion — on six pooled acted-speech corpora, and decomposes the pooled score into speaker-overlap inflation, corpus-specific structure and run-to-run variation.

## Repository content

```
notebooks/
  ser_multicorpus_pipeline.ipynb   all experiments of the paper (single notebook)
requirements.txt
CITATION.cff
LICENSE
```

The notebook is organised in three parts:

| Part | Cells (sections) | Paper |
|---|---|---|
| **Reference pipeline** | Sections 0–21: manifest, label harmonisation, duration audit, feature caching, model, class-balanced focal loss, five-fold speaker-disjoint CV, baselines, protocol-sensitivity split, LOCO, corpus probe, table export | Sections 5.1–5.6 and 5.11 (primary evidence) |
| **Cache-accelerated pipeline** | Diagnostic and Sections 1–6 after the first separator: cached 13-layer WavLM states, ablations, run-to-run variation, frozen vs. fine-tuned WavLM, EMODB leave-one-speaker-out | Sections 5.7–5.10 (supporting evidence) |
| **V10 multi-encoder extension** | Sections V10-1 to V10-6 (WavLM + HuBERT + wav2vec 2.0) | **Not reported in the paper** — exploratory follow-up work |

## Data

The six corpora are public but are **not redistributed** here; access is subject to each dataset's licence. The notebook was run on Kaggle and discovers the corpora automatically under `/kaggle/input`. The Kaggle copies used were:

| Corpus | Kaggle dataset |
|---|---|
| CREMA-D | `ejlok1/cremad` |
| EMODB | `piyushagni5/berlin-database-of-emotional-speech-emodb` |
| MLEnd Spoken Numerals | `jesusrequena/mlend-spoken-numerals` |
| RAVDESS | `uwrfkaggler/ravdess-emotional-speech-audio` |
| SAVEE | `ejlok1/surrey-audiovisual-expressed-emotion-savee` |
| TESS | `ejlok1/toronto-emotional-speech-set-tess` |

To run elsewhere, set the corpus folders in `USER_PATHS` (Section 2) and change `cache_dir` / `out_dir` in `CFG` (Section 0).

## Running

1. On Kaggle, create a notebook with a GPU (T4 was used), attach the six datasets above and upload `ser_multicorpus_pipeline.ipynb`.
2. Install dependencies if needed (`pip install -r requirements.txt`; on Kaggle only `transformers`, `librosa` and `soundfile` may be missing).
3. Run all cells of the reference pipeline. The experiment scheduler (Section 19) runs the plan within `CFG["budget_hours"]` and can resume an interrupted plan.
4. Result tables are exported as LaTeX to `<out_dir>/tables/` (Section 20).

**Settings used for the reported results.** The figures in the paper were obtained with the default preset `CFG["preset"] = "quick"` (at most 18 epochs, early-stopping patience 5), seed 42, and supplementary seeds 123 and 456 for the first two folds. The `"paper"` preset (60 epochs, patience 12) is a longer configuration that was **not** used for the reported figures. The `"smoke"` preset is a quick functional test.

The cache-accelerated pipeline sets no training seed, so its runs are independent but not exactly re-creatable; this is the intended source of the run-to-run variation reported in the paper (three identical runs spanning 2.46 UAR points).

## Key results (paper)

| Quantity | Value |
|---|---|
| Five-fold speaker-disjoint UAR, principal level (CREMA-D, EMODB, RAVDESS, SAVEE) | 68.03% (95% CI 64.95–70.91) |
| UAR inflation from a random speaker-overlapping split | +8.87 (principal) / +18.28 (with TESS) points |
| UAR drop when a corpus is withheld from training | 15–50 points for four of six corpora |
| Corpus identity decoded from the fused representation | 83.51% (majority baseline 48.15%) |
| Range of three identical runs | 2.46 UAR points |

## Citation

If you use this code, please cite the paper (see `CITATION.cff`; the reference will be updated after publication).

## Licence

Code released under the MIT licence (see `LICENSE`). The speech corpora keep their own licences.
