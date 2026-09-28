# CardioGPT: a cardiology expert GPT, pre-trained from scratch

Two GPT language models pre-trained from random weights, in pure PyTorch, on 66.6 million tokens of
cardiology text crawled from PubMed, Wikipedia and NIH, CDC and MedlinePlus. Both share an identical
10.6M-parameter Transformer and differ only in the tokenizer: GPT-2 BPE through **tiktoken**, or
single characters (**letters**).

*Ashish Julakanti · ITS 52000 Applied Machine Learning · Purdue University Northwest · Fall 2026*

Full write-up: [`docs/CardioGPT_Report.pdf`](docs/CardioGPT_Report.pdf) · Slides: [`docs/CardioGPT_Slides.pdf`](docs/CardioGPT_Slides.pdf)

## Results

| | Letters model | tiktoken model |
|---|---|---|
| Validation bits per character (lower is better) | 1.046 | **0.791** |
| Cardiology quiz, 46 questions (random guessing: 25%) | 56.5% | **71.7%** |
| Parameters, total / Transformer body | 10.7M / 10.6M | 29.9M / 10.6M |
| Training on one NVIDIA T4 | 9,836 steps, 3.3 h, 1 pass | 4,463 steps, 2.1 h, 4 passes |

On held-out test documents the tiktoken model is 19–31% lower in bits per character on every
source. On 8 out-of-domain control questions both models fall toward chance, so their knowledge
comes from the cardiology data. All numbers: [`reports/eval_results.md`](reports/eval_results.md).

The tiktoken model's final answer to *"What is atrial fibrillation?"*:

> Atrial fibrillation is a form of arrhythmia within the heart, which is a medical condition that
> affects one or two of the chambers of your heart.

Fluent, domain-specific, and often wrong in the details: **not medical advice.**

## Data (HW1)

| Source | Collected | After cleaning | GPT-2 tokens |
|---|---|---|---|
| PubMed abstracts (NCBI E-utilities API) | 149,890 | 148,223 | 61.1M |
| Wikipedia articles (MediaWiki API, category crawl) | 3,109 | 2,197 | 3.2M |
| NIH, CDC and MedlinePlus pages (focused web crawler) | 1,022 | 844 | 1.0M |
| Question-answer pairs built from section headings | | 8,053 | 1.3M |
| **Total** | **154,021** | **159,317** | **66.6M** |

Cleaning: HTML extraction and normalisation, Gopher-style quality filter, topic filter, exact
deduplication and MinHash-LSH near-duplicate removal (written from scratch). Details:
[`reports/cleaning_funnel.md`](reports/cleaning_funnel.md) and
[`reports/data_stats.md`](reports/data_stats.md).

## Rules followed

- **PyTorch only:** the model and training use only `torch` and `numpy`; no HuggingFace or other ML library.
- **Pre-training only:** both models start from random weights and train in a single run; there is no fine-tuning step.
- **tiktoken and letters:** both tokenizers train the same Transformer and are compared in bits per character.
- **Scraped and cleaned data:** three crawlers written for this project, with a measured cleaning pipeline.

## Repository layout

```
config.py                  specialty settings: queries, categories, seed sites, keywords
scrape/                    PubMed harvester, Wikipedia crawler, focused web crawler
clean/                     HTML extraction, quality filters, MinHash-LSH dedup, Q&A pairs
data_stats.py              corpus statistics and figures
tokenizer.py               character tokenizer and tiktoken wrapper
prepare.py                 splits, mixture weights, binary token files
model.py                   the GPT, written from scratch (RoPE, RMSNorm, SwiGLU, KV cache)
train.py                   training (mixed precision, gradient accumulation, cosine schedule, multi-GPU)
evaluate.py                test bits per character, knowledge quiz, embedding neighbours
ask.py                     ask the models questions
visualize.py               training curves, attention maps, learning progress
tests/test_all.py          8 correctness tests (causality, KV cache, attention, ...)
evals/                     the 54-question quiz and the open questions
reports/                   all results as markdown, plus figures/
runs/                      training logs of both models (checkpoints not included)
CardioGPT_pipeline.ipynb   the Colab notebook that produced these results
docs/                      report and slides (PDF)
```

## Reproduce

```bash
pip install -r requirements.txt
# put your own e-mail into CONTACT_EMAIL in config.py (Wikipedia and NCBI require it)
python tests/test_all.py

python -m scrape.pubmed
python -m scrape.wikipedia
python -m scrape.web_crawler
python -m clean.clean --workers 2
python data_stats.py
python prepare.py --tokenizer char
python prepare.py --tokenizer gpt2

python train.py --tokenizer char --preset char-small --out_dir runs/char-small
python train.py --tokenizer gpt2 --preset bpe-small --batch_size 16 --grad_accum 8 --out_dir runs/bpe-small

STAGES="eval" bash run_all.sh
python ask.py --ckpt runs/char-small runs/bpe-small -q "What is atrial fibrillation?"
```

Crawling takes about 40 minutes; training takes about 5.5 GPU-hours on a T4.

**Full project download:** the complete project folder, including the crawled and tokenized data
and both trained model checkpoints (912 MB), is attached to the
[v1.0 release](../../releases/tag/v1.0) of this repository.

## Data licensing

NIH, CDC and MedlinePlus pages are US-government works; MedlinePlus's licensed encyclopedia and
drug pages were excluded. Wikipedia text is CC BY-SA 4.0. PubMed abstracts are used for education.
The crawlers identify themselves, obey robots.txt and stay within each site's rate limits.
