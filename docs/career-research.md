# Career and project research

Research date: 19 November 2026. This is a targeted scan, not a statistical survey of all jobs. Listings can close or change.

## What the resume already demonstrates

The resume shows experience in PyTorch, recurrent forecasting, CPU inference, custom INT8 quantization, scientific baselines, benchmark leakage analysis, and chemistry LLM data generation. It reports a 28th-place challenge finish and a 75% model-memory reduction. The genomics work is particularly strong evidence of scientific judgment because it reports a negative result and a shortcut rather than overstating performance.

The main evidence gap is the path from experiment to an operable service: a public, reproducible evaluation, explicit train/test boundaries, baseline comparisons, inference latency and reliability measurements, and monitoring. This is a gap in the document, not a finding that the candidate lacks these skills. There is also little evidence of sustained users, cloud deployment, distributed training, or SQL in an ML pipeline. None should be claimed without evidence.

Resume improvements:

1. Tailor the summary and project order to each role: efficient forecasting for ML systems, leakage analysis for research, chemistry agents for applied AI.
2. Give a reader enough context to interpret metrics: dataset, split, baseline, sample size, and whether the result is independently reproducible. Link code or a report where possible.
3. Replace broad tool lists with the tools used in the strongest projects. Add shipped-service and benchmark evidence only after it exists.
4. Keep exploratory work and targets labeled as such. Avoid presenting synthetic or offline results as validated industrial performance.
5. Consider a focused one-page version for internship applications; retain a detailed research version for labs. Confirm graduation and work-authorization requirements for each role.

## Hiring signals

| Evidence | Signal | Practical implication |
|---|---|---|
| [Stanford AI Index 2026 economy chapter](https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_4_economy.pdf) | Tracks AI skills in US job postings, including generative AI. It is aggregate evidence, not a ranking of individual portfolio projects. | Keep generative AI in the portfolio, but do not infer that every employer wants the same project. |
| [Enterpret ML intern](https://job-boards.greenhouse.io/enterpret/jobs/7723230003) | Python, retrieval/agents/evaluation, testable code, and rigorous evaluation of nontrivial LLM work. | Existing chemistry agents can be presented with stronger evaluation and artifacts. |
| [AppLovin ML engineering intern](https://job-boards.greenhouse.io/applovin/jobs/4713038006) | Python, ML fundamentals, real data, and end-to-end ownership from framing to offline evaluation. | Prioritize a complete system with an honest holdout and a simple baseline. |
| [Workday ML engineer intern](https://workday.wd5.myworkdayjobs.com/en-US/Workday_Early_Career/job/Machine-Learning-Engineer-Intern_JR-0110812) | Software and ML engineering; interest in retrieval, embeddings, or model evaluation. | Show both implementation and experimental judgment. |
| [Karya research engineering intern](https://job-boards.greenhouse.io/karya/jobs/5469564008) | Evaluation pipelines, reproducibility, quality control, analysis, and monitoring. | Make failure analysis and checks visible, not just a demo. |

These are examples, not evidence of offer rates or a promise of employment. Job requirements, location, degree and work authorization vary. The candidate's resume already includes agent and RAG skills; another generic chatbot would add less distinct evidence than a measurable deployment project. That is an inference from the resume and the examples above.

## Project decision

| Candidate | Fit | Tradeoff |
|---|---|---|
| Measured forecasting service | Builds directly on recurrent forecasting and CPU constraints; adds API, baselines, temporal evaluation and operational evidence. | A synthetic-data demonstration cannot establish real-world effectiveness. |
| Genomics leakage audit toolkit | Strong research differentiation and a direct extension of existing work. | Meaningful external validation depends on access to appropriate public data and domain-specific labels. |
| Chemistry preference-data evaluation | Extends existing work and current interest in LLM evaluation. | Quality labels and domain validation require careful independent review. |
| Generic document RAG | Relevant to some postings. | Less differentiated for this resume without a domain-specific dataset and rigorous ground truth. |

The selected project is the forecasting service. It makes the missing engineering evidence concrete while building on existing strengths. Follow it with the genomics audit or chemistry evaluation project when suitable independent data are available.

## Evaluation principles

Forecast with observations available at prediction time only; keep later targets outside training. A random split can give misleading estimates for sequential data: see [scikit-learn's lagged-feature example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) and [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Report baseline and model errors on the same held-out period, and describe the data and exact forecast horizon. Measure CPU latency on the actual machine; do not infer a speedup from model size or precision alone.

## How to use this work in applications

Run and understand the repository, reproduce its results, inspect errors and make substantive improvements before describing it as personal work. An honest project bullet can describe the actual service and reported metrics once verified; do not claim production adoption, customer impact, or industrial predictive performance from a synthetic demo.
