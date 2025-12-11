import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# ------------------------------------------------------
# 1. Build dataframes with correct dataset labels
# ------------------------------------------------------

df1 = pd.DataFrame({
    "Test": [f"T{i:02d}" for i in range(1, 16)],
    "M1": [4,3.6,3.8,3.6,4,4,3.8,3.8,3.4,3.4,3.2,3.2,3.8,3.2,2.8],
    "M2": [3.2,3.8,3.8,3,2.2,3.8,2.8,3.4,2.8,3.4,2.8,2.4,3,2.4,2.6],
    "M3": [3.4,3.4,2.8,3.6,3.4,3.6,2.6,3.6,3,3,3.4,2.8,3.2,3.6,3.6],
    "M4": [3.4,3.6,3.4,3.4,2.8,3.4,3,3.2,3.2,3.2,3,2.8,3.2,3.2,2.6]
})
df1["Dataset"] = "Baseline - ChatGPT only"

df2 = pd.DataFrame({
    "Test": [f"T{i:02d}" for i in range(1, 16)],
    "M1": [4.4,3.6,4,4.4,3.2,3,3.4,4.2,4.2,4.2,3,3,2.2,2.2,2.8],
    "M2": [4.8,3.2,3.4,3.6,2.2,4,2.8,3,3.4,4.2,3.4,3.4,3,1.8,2.4],
    "M3": [4,3.6,2.2,3.6,3.6,3.6,3.8,4,3.8,3.6,3.2,3,3,3,2.6],
    "M4": [4,3.2,3.4,3.6,3,3.4,2.8,3.4,3.6,3.8,3.4,3,2.6,1.8,2.6]
})
df2["Dataset"] = "Pipeline - midterm version"

df3 = pd.DataFrame({
    "Test": [f"T{i:02d}" for i in range(1, 16)],
    "M1": [4.6,4.4,4.4,3.6,3.8,4,3.6,3.2,4.6,3.8,4.4,4.8,4.2,4,3.8],
    "M2": [4.8,3.8,4.8,4.4,3.8,4.2,3.2,3.4,4.4,3.6,4.6,4.6,4.6,4,3.6],
    "M3": [4.6,3.6,4.2,4.4,4.2,4.4,3.8,3.6,4.4,3.8,4.8,4.2,4.4,3.6,3.8],
    "M4": [5,4,4.4,4,4,4.4,3.6,3.2,4.2,4.2,4.8,4.4,4.4,3.8,3.8]
})
df3["Dataset"] = "Pipeline - improved #1"

df4 = pd.DataFrame({
    "Test": [f"T{i:02d}" for i in range(1, 16)],
    "M1": [4.6667,4.75,4.25,4.75,5,4.75,4.75,4,4.25,5,4.25,4.25,5,4.5,4],
    "M2": [5,4.75,4.75,4.5,4.75,4.5,4.75,4.25,5,5,4,4,4.75,4,3.75],
    "M3": [4.3333,4.25,3.75,4,4,4.75,4.75,4.75,4.5,5,4.5,4.25,4.25,4.5,4.5],
    "M4": [4.6667,4,3.75,4.25,4,4.75,5,3.75,4,4.75,4.25,4,4.75,4.25,4]
})
df4["Dataset"] = "Pipeline - local LLM (not finetuned)"

df5 = pd.DataFrame({
    "Test": [f"T{i:02d}" for i in range(1, 16)],
    "M1": [5,5,5,4.5,5,5,5,4.5,4,4.5,5,5,5,5,4.5],
    "M2": [5,5,5,4.5,4.5,5,4,4.5,5,4,5,4,4.5,4,4.5],
    "M3": [5,4,4.5,4.5,4,4.5,5,4.5,5,4.5,4.5,4,5,4.5,4.5],
    "M4": [5,5,4.5,4.5,4.5,5,5,4.5,5,4,5,4,5,4.5,4.5]
})
df5["Dataset"] = "Pipeline - improved #2"

# ------------------------------------------------------
# 2. Single unified dataset
# ------------------------------------------------------
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# Compute mean score per test case
df["Mean"] = df[["M1","M2","M3"]].mean(axis=1)

# Enforce temporal order for dataset labels
dataset_order = [
    "Baseline - ChatGPT only",
    "Pipeline - midterm version",
    "Pipeline - improved #1",
    "Pipeline - local LLM (not finetuned)",
    "Pipeline - improved #2"
]
df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)

# ------------------------------------------------------
# 3. Boxplot (metric distributions)
# ------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df.melt(id_vars="Dataset", value_vars=["M1","M2","M3"]),
    x="variable", y="value"
)
plt.title("Distribution of Metrics Across All Datasets")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.savefig("boxplot_metrics.png", dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------------------------------
# 4. Bar chart (mean metrics per dataset)
# ------------------------------------------------------
metric_means = df.groupby("Dataset")[["M1","M2","M3"]].mean()

plt.figure(figsize=(12, 6))
metric_means.loc[dataset_order].plot(kind="bar")
plt.title("Average Metric Scores per Dataset")
plt.ylabel("Mean Score")
plt.xticks(rotation=20)
plt.savefig("barplot_dataset_means.png", dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------------------------------
# 5. Evolution of overall performance
# ------------------------------------------------------
dataset_order = [
    "Baseline - ChatGPT only",
    "Pipeline - midterm version",
    "Pipeline - improved #1",
    "Pipeline - local LLM (not finetuned)",
    "Pipeline - improved #2"
]

avg_by_dataset = df.groupby("Dataset")["Mean"].mean().loc[dataset_order]

plt.figure(figsize=(10, 6))
sns.lineplot(x=dataset_order, y=avg_by_dataset.values, marker="o")
plt.title("Evolution of Overall Pipeline Quality")
plt.ylabel("Mean Score")
plt.ylim(0, 5)
plt.xticks(rotation=20)
plt.savefig("lineplot_pipeline_evolution.png", dpi=300, bbox_inches='tight')
plt.close()