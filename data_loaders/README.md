# Data loader variants

During experimentation, each combination of two design choices was implemented
as its own module (a common pattern that grew organically during thesis work).
Renamed here for clarity, they form a 2x3 matrix:

|                        | naive daily aggregation | 20:00 UTC windowed aggregation |
|------------------------|--------------------------|----------------------------------|
| **Daily average**      | `daily_avg.py`           | `daily_avg_windowed.py`          |
| **Daily sum**          | `daily_sum.py`           | `daily_sum_windowed.py`          |
| **Article/row level**  | `article_level.py`       | `article_level_windowed.py`      |

**Aggregation (rows):**
- *Daily average* — sentiment scores are averaged across all articles published on a calendar day.
- *Daily sum* — sentiment scores are summed instead of averaged.
- *Article/row level* — no aggregation; each article is kept as its own row.

**Windowing (columns):**
- *Naive* — an article's "day" is just its calendar date.
- *20:00 UTC windowed* — an article is assigned to the trading day whose 20:00 UTC
  close it should influence: Mon-Thu before 20:00 UTC counts same-day, at/after
  20:00 UTC rolls to the next day; Friday at/after 20:00 UTC and all weekend
  activity rolls to the following Monday. This models the fact that news
  arriving after market close shouldn't be treated as same-day signal.

All six variants expose the same `get_datasets(...)` interface (see each
module) and were originally selected by editing a single import line at the
top of the training notebook, e.g.:

```python
from data_loaders.daily_avg import get_datasets
# from data_loaders.daily_avg_windowed import get_datasets
# from data_loaders.daily_sum import get_datasets
# from data_loaders.daily_sum_windowed import get_datasets
# from data_loaders.article_level import get_datasets
# from data_loaders.article_level_windowed import get_datasets
```

That toggle pattern is preserved in `notebooks/main_training_pipeline.ipynb`
and `notebooks/baseline_no_tuning.ipynb`.
