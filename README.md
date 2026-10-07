# SHL Hiring Assessment 2026 - Grammar Scoring Engine

Predicts a grammar score (0-5) from 45-60 second spoken audio clips.

## Approach
1. Speech to text with Whisper (small).
2. Text features: GPT-2 perplexity, sentence statistics, repetition and filler ratios, MiniLM sentence embeddings.
3. Audio features: wav2vec2 embeddings (layers 6 and 12, mean pooled).
4. Ridge regression on all features, selected by 5-fold cross-validation.

## Results
| Metric | RMSE |
|---|---|
| Baseline (predict the mean) | 1.2382 |
| Train | 0.3994 |
| 5-fold CV | 0.596 |
| Kaggle public leaderboard | 0.4592 |

## Notes
- The train and test folders contain files with identical names, so file paths are built with the split folder.
- Run on Kaggle with GPU (T4) and internet on.

## Files
- `shl_grammar_scoring_v3.ipynb`: full notebook with code, report and results.
- `submission.csv`: predictions for the test set.
