# Cereal Nutrition Analysis

An exploratory data analysis (EDA) project on cereal nutrition facts — applying
pandas fundamentals, correlation analysis, and matplotlib/seaborn visualization
to understand how nutritional content varies by manufacturer.

## Project Structure

```
.
├── data
│   └── cereal.csv              # Raw dataset (30 cereals x 10 nutrition columns)
├── images
│   ├── 01_univariate_bar.png   # Count of cereals per manufacturer
│   ├── 02_sorted_bar.png       # Average calories per manufacturer (sorted)
│   ├── 03_heatmap.png          # Correlation heatmap across all nutrition facts
│   ├── 04_scatter.png          # Sugar vs. calories (bivariate)
│   └── 05_multivariate_scatter.png  # Sugar vs. calories, colored by manufacturer
├── models                      # Reserved for future predictive modeling
├── README.md
└── sample.ipynb                # Full analysis notebook (run top to bottom)
```

## Workflow

The notebook (`sample.ipynb`) follows this sequence:

1. **Load** the data from `data/cereal.csv`
2. **Inspect** shape, dtypes, and summary statistics
3. **Index** on cereal name
4. **Filter** with boolean masks (e.g., low-sugar Kelloggs cereals)
5. **Aggregate** with `groupby` (average calories and cereal count per manufacturer)
6. **Correlate** — pairwise correlation and a full correlation matrix
7. **Visualize** — univariate, bivariate, and multivariate charts (saved to `images/`)
8. **Interpret** the findings, with a reminder that correlation ≠ causation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas matplotlib seaborn jupyter
```

## Usage

```bash
jupyter notebook sample.ipynb
```

Run all cells top to bottom. Charts are regenerated and overwritten in `images/`
on every run, so the notebook is always the source of truth for the figures.

## Key Findings

- **Sugar and calories are positively correlated** — cereals higher in sugar
  tend to be higher in calories (see `images/04_scatter.png`). This is a
  correlation, not evidence that one causes the other.
- Average calories per serving vary noticeably by manufacturer
  (see `images/02_sorted_bar.png`).
- The full pairwise correlation matrix (`images/03_heatmap.png`) is useful for
  exploration but is intentionally **not** used as a stakeholder-facing chart —
  raw heatmaps are hard for a non-technical audience to read.

## Next Steps

- `models/` is reserved for a future predictive task on this dataset
  (e.g., predicting a cereal's nutrition "rating" from its ingredient profile).
- Consider expanding `data/` with a larger cereal dataset (Kaggle's "80 Cereals"
  dataset is a natural fit) to get more statistically meaningful correlations.

## License

# Python for Data Science — Mastery Guide
*Based on: Python for Data Science, Part 1 — Dr Ahmed Toujani (Sep 2025)*

---

## Table of Contents
1. [What is Data Science?](#1-what-is-data-science)
2. [AI vs ML vs DL vs DS](#2-ai-vs-ml-vs-dl-vs-ds)
3. [Data Analytics vs Data Science](#3-data-analytics-vs-data-science)
4. [The Data Science Lifecycle](#4-the-data-science-lifecycle)
5. [Skills You Need](#5-skills-you-need)
6. [Python Objects & Classes](#6-python-objects--classes)
7. [Working in Colab](#7-working-in-colab)
8. [Reading & Debugging Errors](#8-reading--debugging-errors)
9. [Getting Data into Colab](#9-getting-data-into-colab)
10. [Loading Data with Pandas](#10-loading-data-with-pandas)
11. [Headers](#11-headers)
12. [Inspecting a DataFrame](#12-inspecting-a-dataframe)
13. [Checking Data Types](#13-checking-data-types)
14. [Selecting Columns](#14-selecting-columns)
15. [Rows & Columns Count](#15-rows--columns-count)
16. [Index Management](#16-index-management)
17. [Filtering Data (Boolean Masks)](#17-filtering-data-boolean-masks)
18. [Combining Filters](#18-combining-filters)
19. [Locating Min/Max Rows](#19-locating-minmax-rows)
20. [Exploratory Data Analysis (EDA)](#20-exploratory-data-analysis-eda)
21. [Data Visualization Libraries](#21-data-visualization-libraries)
22. [Univariate Visualizations](#22-univariate-visualizations)
23. [Correlation](#23-correlation)
24. [Scatter Plots](#24-scatter-plots)
25. [Bar Charts (with Groupby)](#25-bar-charts-with-groupby)
26. [Multivariate Analysis (3rd Variable)](#26-multivariate-analysis-3rd-variable)
27. [Choosing the Right Chart](#27-choosing-the-right-chart)
28. [Explanatory vs Exploratory Graphs](#28-explanatory-vs-exploratory-graphs)
29. [Case Study: Schooling & Life Expectancy](#29-case-study-schooling--life-expectancy)
30. [Quick-Reference Cheat Sheet](#30-quick-reference-cheat-sheet)

---

## 1. What is Data Science?

> "Data science is a multidisciplinary field that involves extracting meaningful insights and knowledge from large and complex sets of data. It combines techniques from various domains, including statistics, mathematics, computer science, and domain expertise, to uncover patterns, make predictions, and gain valuable insights from data."

Data Science sits at the intersection of:
- **Mathematics**
- **Machine Learning**
- **Computer Science**
- **Statistical Research**
- **Data Processing**
- **Domain Expertise**

**Mastery tip:** Data science is not just coding — it's coding + stats + business/domain context, applied together to answer real questions.

---

## 2. AI vs ML vs DL vs DS

| Term | Definition |
|---|---|
| **AI** (Artificial Intelligence) | Programs with the ability to learn and reason like humans (the broadest goal) |
| **ML** (Machine Learning) | Algorithms with the ability to learn **without being explicitly programmed** |
| **DL** (Deep Learning) | A **subset of ML** where artificial neural networks adapt and learn from vast amounts of data |
| **DS** (Data Science) | Sits at the overlap of AI/ML/DL and Math & Statistics/Visualization/EDA |

**Mental model (nesting):**
```
AI ⊃ ML ⊃ DL
DS = overlap of (AI/ML/DL) and (Math, Stats, Visualization, EDA)
```

---

## 3. Data Analytics vs Data Science

| Type | Question Answered | Direction |
|---|---|---|
| **Descriptive Analytics** | What happened? | Hindsight |
| **Diagnostic Analytics** | Why did it happen? | Hindsight |
| **Predictive Analytics** | What *will* happen? | Foresight |
| **Prescriptive Analytics** | How can we *make* it happen? | Foresight |

**Key takeaway:** Data Science adds the **predictive layer** on top of the **descriptive layer** of Data Analytics.

---

## 4. The Data Science Lifecycle

A 7-step cyclical process:

1. **Business Understanding** — Ask relevant questions; define the objective.
2. **Data Mining** — Gather and scrape the data needed for the project.
3. **Data Cleaning** — Fix inconsistencies; handle missing values.
4. **Data Exploration** — Form hypotheses by visually analyzing the data.
5. **Feature Engineering** — Select/construct meaningful features from raw data.
6. **Predictive Modeling** — Train ML models, evaluate performance, make predictions.
7. **Data Visualization** — Communicate findings with plots/interactive visuals.

**Mastery tip:** This is a *cycle*, not a straight line — insights from step 7 often send you back to step 1.

---

## 5. Skills You Need

The "Data Science Minimum" includes:
- Math & Statistics
- Ethical Skills
- Team Player Skills
- Lifelong Learning Skills
- Communication Skills
- Real World Project Skills
- Machine Learning Skills
- Data Visualization Skills
- **Data Wrangling & Preprocessing**
- **Coding Skills (Python, R)**

This course focuses on the **first three practical steps**: Coding → Data Wrangling → Data Visualization.

---

## 6. Python Objects & Classes

- **Everything in Python is an object.**
- Every object is an **instance** of a blueprint called a **class**.

| Concept | Meaning | Example |
|---|---|---|
| **Class** | The *type* of object (blueprint) | `string`, `list`, `tuple` |
| **Object** | A *specific instance* of a class | `'hello world'`, `[1,2,3]`, `('hello','world')` |

**How much do you need to know about classes as a Data Scientist?**
- Enough to understand **how to use them** (e.g., NumPy arrays, Pandas DataFrames).
- You do **NOT** need to write your own classes.

---

## 7. Working in Colab

When opening a shared Colab notebook:
1. Open the **File** menu
2. **Save a Copy in Drive**
3. Change it, play with it, break it, fix it — make it your own.

---

## 8. Reading & Debugging Errors

**Golden rule: Listen to your error messages!** The most useful info is usually at the **bottom** of the traceback (scroll down if needed).

| Error Type | What it Means | Example Cause |
|---|---|---|
| **SyntaxError** (`unexpected EOF while parsing`) | Missing a closing bracket/quote — `()`, `[]`, `{}` not balanced | `print('hi'` (missing closing `)`) |
| **NameError** (`name 'X' is not defined`) | You referenced a variable that doesn't exist | Case sensitivity typo (`Y` vs `y`), or forgot to import a library and use its alias (e.g., `pd`) |
| **SyntaxError** (`invalid syntax`) | Typo in code structure | Quotes both at the end instead of one at start/end; misspelled keyword (`pritn` instead of `print`) |

**Debugging checklist:**
1. Read the bottom line of the traceback first.
2. Check for unclosed brackets/quotes.
3. Check variable name spelling & case sensitivity.
4. Check that required libraries are imported with the correct alias.

---

## 9. Getting Data into Colab

**Three options:**

**Option 1 — Upload to temporary Colab environment**
1. Download the data locally
2. Open the file folder icon on the left sidebar
3. Drag and drop the file to upload
4. Right-click the file → "Copy path"

**Option 2 — Save file in Google Drive**
1. Mount your Drive
2. Open the file folder in Colab
3. Navigate to the file in Drive
4. Right-click → "Copy path"

**Option 3 — Publish a shared web address (Google Sheets)**
1. Open the Google Sheets file
2. File → Share → Publish to Web
3. Select the sheet/document from the dropdown
4. Select **CSV** as the format
5. Click Publish → Confirm → Copy the link

---

## 10. Loading Data with Pandas 🐼

Different file types need different loading functions:

```python
import pandas as pd

df = pd.read_csv('filepath')     # for .csv files
df = pd.read_excel('filepath')   # for .xlsx files
```

- `'filepath'` can be a **remote URL** as long as it points **directly** to a `.csv` or `.xlsx` file.
- Other pandas functions exist for other file types (JSON, SQL, etc.), but these two are the workhorses.

---

## 11. Headers

Pandas assumes the **top row is the header** by default. You can override this:

| Situation | Code |
|---|---|
| Header is the top row (default) | `pd.read_excel('path')` or `pd.read_excel('path', header=0)` |
| No header at all | `pd.read_excel('path', header=None)` |
| Header present but not at the top (e.g., title rows above it) | `pd.read_excel('path', header=2)` *(index of the actual header row)* |

---

## 12. Inspecting a DataFrame

```python
df.head()      # first 5 rows
df.head(8)     # first 8 rows
df.tail()      # last 5 rows
df.tail(11)    # last 11 rows
df.sample(5)   # 5 random rows
```

---

## 13. Checking Data Types

```python
df.info()             # data types + index + column names + non-null counts
df.dtypes             # data type for ALL columns
df['name'].dtypes      # data type for ONE column
```

---

## 14. Selecting Columns

```python
df['name']                       # 1 column → returns a Pandas Series
df[['name']]                     # 1 column → returns a Pandas DataFrame
df[['name', 'Manufacturer']]     # multiple columns → DataFrame
```

**Mastery tip:** Double square brackets `[[ ]]` = you get a DataFrame back (keeps 2D structure). Single brackets `[ ]` = you get a Series (1D).

---

## 15. Rows & Columns Count

```python
df.info()           # includes shape-related info
df.shape             # (rows, columns) tuple
len(df)              # number of rows
len(df.columns)      # number of columns
```

---

## 16. Index Management

- When you load a dataset, pandas auto-assigns a **0-based index** to each row.
- You can replace it with a meaningful column (e.g., `Employee_ID`) — but every value in that column **must be unique**.

```python
df.set_index(['Employee_ID'], drop=True, inplace=True)
```
- `drop=True` → the original column is removed after becoming the index.
- `inplace=True` → permanently changes `df` (no need to reassign).

---

## 17. Filtering Data (Boolean Masks)

String methods that return booleans are great for filtering:

```python
df['City'].str.startswith('B')     # cities starting with 'B'
df['City'].str.endswith('i')       # cities ending in 'i' (e.g., Miami)
df['City'].str.contains('or')      # cities containing 'or' (e.g., New York)
```

---

## 18. Combining Filters

Use these operators to combine multiple boolean filters:

| Symbol | Meaning |
|---|---|
| `&` | and |
| `\|` | or |
| `~` | not |

```python
RI_filter = df['State'] == 'RI'
MI_filter = df['State'] == 'MI'
Pop_filter = df['Total_Population'] > 10000
B_filter = df['City'].str.startswith('B')

# Combine: all cities in RI with population > 10000
df[RI_filter & Pop_filter]
```

**Important:** No output ≠ error. It just means **no rows meet all your criteria**.

---

## 19. Locating Min/Max Rows

```python
# Option 1: two steps
min_value = df['column'].min()
df.loc[df['column'] == min_value, :]

# Option 2: one line
df.loc[df['column'] == df['column'].min(), :]
```

---

## 20. Exploratory Data Analysis (EDA)

Two guiding questions:

**1. What is your dataset *like*?**
```python
df.shape
df.describe()
```

**2. How is the data *distributed*?**

| Data Type | Question | Tool |
|---|---|---|
| Categorical | How many of each category? | Bar chart |
| Numeric | What values are most of the data near? | Histogram |
| Numeric | Where are the outliers? | Box plot |

---

## 21. Data Visualization Libraries

"The Power behind the Plotting":
- **`matplotlib.pyplot`** (imported as `plt`)
- **`seaborn`** (imported as `sns`)
- **Pandas built-in plotting** (`.plot()`)

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 22. Univariate Visualizations

= exploring **one column at a time**.

Example: a horizontal bar chart of "Number of Cereals by Manufacturer" — showing counts per category.

---

## 23. Correlation

**Definition:** When two or more variables tend to change together.
- One goes up → the other tends to go up (**positive correlation**)
- One goes up → the other tends to go down (**negative correlation**)

### Correlation Coefficient (r)
- Ranges from **-1 to +1**
- Sign (+/-) = **direction**
- Magnitude = **strength**
- Closer to 0 = weaker correlation

```
-1 ―――――――――――――――― 0 ―――――――――――――――― +1
Strong negative | Weak negative | Weak positive | Strong positive
```

**Strength scale:**
| Range | Strength |
|---|---|
| 0.3 – 0.5 | Low |
| 0.5 – 0.7 | Moderate |
| > 0.7 | Strong |

```python
correlation = cereal['calories per serving'].corr(cereal['grams of sugars'])
# e.g., 0.5623 → moderate positive correlation
```

### ⚠️ Correlation ≠ Causation
Classic example: "Number of people who drowned falling into a pool" correlates 66.6% with "Films Nicolas Cage appeared in" — obviously not causal! Always be skeptical before claiming X causes Y.

### Correlation Matrix & Heatmap

```python
cereal.corr()   # correlation matrix for ALL numeric columns

import seaborn as sns
corr = cereal.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, cmap='Blues', annot=True)
```

---

## 24. Scatter Plots

A scatter plot maps each data point as `(x, y)` on a Cartesian plane — the best visualization for **correlation**.

| Pattern | Meaning |
|---|---|
| Points trend up-right | Positive correlation |
| Points trend down-right | Negative correlation |
| Points scattered with no trend | No correlation |

```python
cereal.plot.scatter(x='grams of sugars', y='calories per serving')
```
**Interpretation example:** Cereals with more sugar tend to have more calories (positive correlation) — and vice versa.

### Color-coding a 3rd variable in a scatterplot
```python
sns.scatterplot(data=cereal, x='calories per serving', y='grams of sugars', hue='type')
plt.legend(bbox_to_anchor=(1,1))
```

---

## 25. Bar Charts (with Groupby)

```python
manu_cal = cereal.groupby('Manufacturer')['calories per serving'].mean()
```

### Vertical Bar Chart
```python
plt.bar(manu_cal.index, manu_cal.values)
plt.ylabel('Average Calories')
plt.xlabel('Manufacturer')
plt.xticks(rotation=30)
plt.show()
```

### Horizontal Bar Chart (`barh`)
```python
plt.barh(manu_cal.index, manu_cal.values)
plt.ylabel('Manufacturer')
plt.xlabel('Average Calories')
plt.show()
```
**Mastery tip:** Horizontal bar charts are especially useful when category labels are **long** (they don't get cut off/rotated).

### Sorted Bar Chart (best practice for readability)
```python
manu_cal = manu_cal.sort_values()
plt.figure(figsize=(10,5))
plt.barh(manu_cal.index, manu_cal.values)
plt.show()
```

---

## 26. Multivariate Analysis (3rd Variable)

= exploring relationships among **multiple variables** (most commonly **bivariate** = 2 variables at once, or trivariate with a 3rd dimension via color/`hue`).

```python
manufact = cereal.groupby(['Manufacturer', 'type']).mean().reset_index()
manufact = manufact.sort_values(by='calories per serving')

sns.barplot(data=manufact, x='calories per serving', y='Manufacturer', hue='type')
plt.title('Average Calories in Cereal by Manufacturer and Type')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
```
The `hue` parameter injects a **3rd categorical variable** into 2D plots (bar charts, scatter plots) via color coding.

---

## 27. Choosing the Right Chart

| Data Relationship | Chart Type |
|---|---|
| Correlation | Scatter Plot |
| Distribution | Histogram, Box Plot |
| Geospatial | Maps |
| Ranking / Comparison | Bar Chart |
| Trends (over time) | Line Graph |

**Resources for inspiration:**
- Selecting visualizations: python-graphgallery.com, datavizproject.com
- Example galleries: seaborn.pydata.org/examples, matplotlib.org/stable/gallery

---

## 28. Explanatory vs Exploratory Graphs

When creating **explanatory** graphs (i.e., for an audience, not just yourself), **avoid**:
- ❌ Box Plots
- ❌ Correlation Heatmaps
- ❌ Unlabeled Pie Charts

**Use instead:**
- ✅ Histograms
- ✅ Bar Graphs
- ✅ Scatterplots
- ✅ Line Graphs
- ✅ Well-Labeled Pie Charts

**Why:** Exploratory tools (box plots, heatmaps) are great for *you* to discover patterns, but they're often too technical/dense for a general audience. Simplify for communication.

---

## 29. Case Study: Schooling & Life Expectancy

**Dataset:** Country-level statistics across 15 years (one row = one country/year), sourced from Kaggle.

### Question 1: Does education increase life expectancy?
- **Method:** Scatter plot (Schooling Index vs. Life Expectancy), color-coded by life expectancy.
- **Finding:** Clear **positive correlation** — nations with higher schooling tend to have longer life expectancy.
- **Caveat:** Correlation ≠ causation — can't conclude schooling *causes* longer life.

### Question 2: Does development status affect schooling or life expectancy?
- **Method:** Bar charts comparing "Developed" vs. "Developing" countries on both metrics.
- **Finding:** Developed countries have **higher average life expectancy AND higher average schooling index**.
- **Interpretation:** May indicate developed countries invest more in education, or that education drives development — direction of causality is unclear.

### Question 3: How have schooling and life expectancy changed over time (2000–2014)?
- **Method:** Line graphs of both metrics across years.
- **Finding:** Both **increased together** year-over-year.
- **Interpretation/Recommendation:** Since the two trends move together, schooling access may function as a protective factor (e.g., a "safe haven" reducing mortality) — leading to a recommendation to **improve access to safe schooling** to help raise life expectancy.

**Mastery tip:** This case study models the full analysis workflow: ask a question → pick the right visualization → interpret cautiously → make a data-backed recommendation (while flagging correlation/causation limits).

---

## 30. Quick-Reference Cheat Sheet

### Imports
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### Load Data
```python
df = pd.read_csv('path')
df = pd.read_excel('path', header=0)
```

### Inspect
```python
df.head() / df.tail() / df.sample(5)
df.info() / df.dtypes / df.shape
df.describe()
len(df) / len(df.columns)
```

### Select & Index
```python
df['col']                      # Series
df[['col1','col2']]            # DataFrame
df.set_index(['ID'], drop=True, inplace=True)
```

### Filter
```python
mask = df['col'] > 10
df[mask]
df[mask1 & mask2]              # AND
df[mask1 | mask2]              # OR
df[~mask1]                     # NOT
df['col'].str.startswith('B') / .str.endswith('i') / .str.contains('or')
```

### Aggregate
```python
df.groupby('cat_col')['num_col'].mean()
df.loc[df['col'] == df['col'].min(), :]
```

### Correlation
```python
df['a'].corr(df['b'])
df.corr()
sns.heatmap(df.corr(), cmap='Blues', annot=True)
```

### Plot
```python
df.plot.scatter(x='a', y='b')
plt.bar(x, y) / plt.barh(x, y)
plt.xlabel() / plt.ylabel() / plt.title() / plt.xticks(rotation=30)
sns.scatterplot(data=df, x='a', y='b', hue='cat')
sns.barplot(data=df, x='a', y='b', hue='cat')
plt.legend(bbox_to_anchor=(1,1))
plt.show()
```

### Error-Reading Checklist
1. Read the **bottom** of the traceback.
2. `SyntaxError` → unclosed bracket/quote or typo'd keyword.
3. `NameError` → undefined variable, case mismatch, or missing import/alias.

---

*End of Mastery Guide — review each section, then try re-deriving the case study (Section 29) from scratch using a new dataset to confirm mastery.*