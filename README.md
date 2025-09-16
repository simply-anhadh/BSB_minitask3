:

🌸 Iris Dataset Visualization & Analysis
📂 Project Structure

iris.csv – dataset (150 rows, 5 columns)

iris_analysis_fixed.py – Python script for analysis & plots

summary_by_species.csv – summary statistics

Plots (PNG files):

plot_petal_scatter.png

boxplot_sepal_length.png

boxplot_sepal_width.png

boxplot_petal_length.png

boxplot_petal_width.png

violin_petal_length.png

pairwise_scatter_matrix.png

📊 Dataset Description

The Iris dataset is a well-known dataset introduced by Ronald Fisher.
It consists of 150 flower samples across three species (Setosa, Versicolor, Virginica) with four features:

Sepal Length

Sepal Width

Petal Length

Petal Width

📈 Visualizations & Insights
1. Scatter Plot – Petal Length vs Petal Width

Shows strong separation of Setosa from other species.

Versicolor and Virginica overlap but differ in ranges.

2. Boxplots – Sepal & Petal Features

Petal length and width clearly differentiate species.

Sepal width overlaps across species, making it less discriminative.

3. Violin Plot – Petal Length Distribution

Distinct clusters: Setosa (small), Versicolor (medium), Virginica (large).

4. Pairwise Scatter Matrix

Petal length & width are strongly correlated.

Sepal measurements alone are weaker indicators.

🚀 How to Run

Clone this repository or download the files.

Install dependencies:

pip install pandas matplotlib scipy seaborn


Run the script:

python iris_analysis_fixed.py


Check the output plots in the project folder.

📌 Key Takeaways

Petal features are the best predictors of species.

Sepal features show more overlap, offering less separation.

Strong correlation between petal length & width aids classification.
