# choice of LLM
looked at https://lmarena.ai/leaderboard/text

## best open source model (but cloud)
the best open source model (as of 19 Oct 2025) is glm-4.6 with an Elo of 1422

command:
```bash
./ollama/bin/ollama run glm-4.6:cloud
```
prompted to sign in to run cloud models, so I did, then execute the above command again
however, this does not run locally. it runs on their cloud.

## for local model
see: https://ollama.com/library/deepseek-r1:latest

DeepSeek R1 performs decently on leaderboard - 11th ranked as of 19 Oct 2025, 1417 Elo
let's use the 5 GB model (8B) which has the "latest" tag.
from DeepSeek docs, it looks like distillation was done on Qwen3-8B to create this model.
Despite being a small model, it performs exceptionally well on benchmarks, 
sometimes even outperforming much larger models such as Qwen3-235B on Humanity's Last Exam.

```bash
./ollama/bin/ollama run deepseek-r1:latest
```

## openAI open-sourced a model, gpt-oss, would be interesting to compare RAG metrics
...


# questions to ask / evals

deep comparison of APH vs NVDA vs AMZN


# scraping from SEC - data quality control

how to check scraping was done correctly?
can't possibly inspect each 10K individually, need a way to automate this

## issues with parsing HTML 
- tried `readability.Document`, but it wrongly discards a lot of relevant content
- safest is simple regex substitution/parsing, however it fails to handle many symbols/special html characters/tags
- final best method is to use `lxml` followed by cleaning of invisible/blank characters and lines
    - we also ensure important SEC section headers are preserved

## challenge: parsing "tables" from HTML and representing them for effective RAG
- even with `lxml`, some tables are difficult to be parsed
- however, tables contain very important financial metric information in 10K filings, and cannot be treated lightly
- since our output is .txt, how should these tables be correctly represented to maximize RAG performance?
    - Markdown?

solution: yes, identify tables via HTML tags -> parse with `pandas` -> generate BOTH Markdown AND list of facts (strings) to enhance RAG performance + facilitate easier citations from these tables.

### example parsed table + fact:
- while imperfect, it should be enough for our purposes

| 0                                                                   | 2                       | 3                       | 4                       | 5                       | 6                       | 7                       | 8                       | 9                       |
|:--------------------------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
|                                                                     | Year Ended December 31, | Year Ended December 31, | Year Ended December 31, | Year Ended December 31, | Year Ended December 31, | Year Ended December 31, | Year Ended December 31, | Year Ended December 31, |
|                                                                     | 2024                    | 2024                    |                         | 2023                    | 2023                    |                         | 2022                    | 2022                    |
| Net income                                                          | $                       | 2441.6                  |                         | $                       | 1945.5                  |                         | $                       | 1916.8                  |
| Total other comprehensive (loss) income, net of tax:                |                         |                         |                         |                         |                         |                         |                         |                         |
| Foreign currency translation adjustments                            |                         | (201.1)                 |                         |                         | (0.9)                   |                         |                         | (265.2)                 |
| Unrealized loss on hedging activities                               |                         | —                       |                         |                         | —                       |                         |                         | (0.1)                   |
| Pension and postretirement benefit plan adjustment                  |                         | 16.6                    |                         |                         | 1.1                     |                         |                         | 11.8                    |
| Total other comprehensive (loss) income, net of tax                 |                         | (184.5)                 |                         |                         | 0.2                     |                         |                         | (253.5)                 |
| Total comprehensive income                                          |                         | 2257.1                  |                         |                         | 1945.7                  |                         |                         | 1663.3                  |
| Less: Comprehensive income attributable to noncontrolling interests |                         | (15.8)                  |                         |                         | (16.3)                  |                         |                         | (9.5)                   |
| Comprehensive income attributable to Amphenol Corporation           | $                       | 2241.3                  |                         | $                       | 1929.4                  |                         | $                       | 1653.8                  |

```
{'caption': '',
 'facts': ['Net income — 2 = $',
  'Net income — 3 = 2441.6',
  'Net income — 5 = $',
  'Net income — 6 = 1945.5',
  'Net income — 8 = $',
  'Net income — 9 = 1916.8',
  'Foreign currency translation adjustments — 3 = (201.1)',
  'Foreign currency translation adjustments — 6 = (0.9)',
  'Foreign currency translation adjustments — 9 = (265.2)',
  'Unrealized loss on hedging activities — 3 = —',
  'Unrealized loss on hedging activities — 6 = —',
  'Unrealized loss on hedging activities — 9 = (0.1)',
  'Pension and postretirement benefit plan adjustment — 3 = 16.6',
  'Pension and postretirement benefit plan adjustment — 6 = 1.1',
  'Pension and postretirement benefit plan adjustment — 9 = 11.8',
  'Total other comprehensive (loss) income, net of tax — 3 = (184.5)',
  'Total other comprehensive (loss) income, net of tax — 6 = 0.2',
  'Total other comprehensive (loss) income, net of tax — 9 = (253.5)',
  'Total comprehensive income — 3 = 2257.1',
  'Total comprehensive income — 6 = 1945.7',
  'Total comprehensive income — 9 = 1663.3',
  'Less: Comprehensive income attributable to noncontrolling interests — 3 = (15.8)',
  'Less: Comprehensive income attributable to noncontrolling interests — 6 = (16.3)',
  'Less: Comprehensive income attributable to noncontrolling interests — 9 = (9.5)',
  'Comprehensive income attributable to Amphenol Corporation — 2 = $',
  'Comprehensive income attributable to Amphenol Corporation — 3 = 2241.3',
  'Comprehensive income attributable to Amphenol Corporation — 5 = $',
  'Comprehensive income attributable to Amphenol Corporation — 6 = 1929.4',
  'Comprehensive income attributable to Amphenol Corporation — 8 = $',
  'Comprehensive income attributable to Amphenol Corporation — 9 = 1653.8'],
 'n_rows': 11,
 'n_cols': 9}
```


## better RAG chunking (already addressed in `lxml` function)
- add light postprocessing to make sure SEC section headers are preserved and become chunk boundaries

example (this is not the code we used in the end, but the idea is similar)
```python3
SEC_SECTION_RE = re.compile(
    r"(?im)^(Item\s+(1A?|2|3|4|5|6|7A?|8|9A?|9B|10|11|12|13|14)\.?\s+[^\n]+)\n"
)
def add_section_breaks(text: str) -> str:
    # Ensure a blank line before and after section headers
    return SEC_SECTION_RE.sub(r"\n\n\1\n\n", text)
```
