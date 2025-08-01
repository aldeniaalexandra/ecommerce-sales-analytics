-----

# E-commerce Sales Analytics Dashboard

A comprehensive business intelligence solution for e-commerce analytics, featuring advanced customer segmentation, market basket analysis, and interactive dashboards.

-----

## Project Overview

This project analyzes **541,909 real e-commerce transactions** from a UK retailer (December 2010 - December 2011). Our goal is to transform raw transactional data into actionable business intelligence through robust data cleaning, advanced analytics, customer segmentation, and an interactive Streamlit dashboard.

### Key Insights

  * **£4.85M** total revenue analyzed, providing a solid foundation for in-depth analytics.
  * **81.69%** data quality retention after comprehensive cleaning, ensuring the reliability of all insights.
  * **7 distinct customer segments** identified through RFM analysis, enabling highly targeted marketing strategies.
  * **76.2%** revenue concentration from the top 20% of products, highlighting key product drivers and areas for focus.
  * **Interactive dashboard** with real-time filtering capabilities, empowering dynamic exploration and visualization of data.

-----

## Quick Start Guide

This guide will help you get the E-commerce Sales Analytics Dashboard up and running directly from GitHub.

### Prerequisites

Before you begin, ensure you have the following software installed on your system:

  * **Python 3.8+**: Download Python from the [official website](https://www.python.org/downloads/).
  * **Kaggle Account**: Required for convenient data acquisition via the Kaggle API. Manual download is also available.
  * **Git**: Download Git from the [official website](https://git-scm.com/downloads/).

### Installation and Setup

1.  **Clone the Repository**:
    Begin by cloning the project repository to your local machine:

    ```bash
    git clone https://github.com/your-username/day-02-ecommerce-dashboard.git
    cd day-02-ecommerce-dashboard
    ```

2.  **Create and Activate Virtual Environment**:
    It's highly recommended to use a virtual environment to manage project dependencies and avoid potential conflicts with other Python projects.

      * **Using `uv` (recommended for speed):**
        ```bash
        uv venv venv-day-02
        # Windows
        venv-day-02\Scripts\activate
        # macOS/Linux
        source venv-day-02/bin/activate
        ```
      * **Using `venv` (built-in Python module):**
        ```bash
        python -m venv venv-day-02
        # Windows
        venv-day-02\Scripts\activate
        # macOS/Linux
        source venv-day-02/bin/activate
        ```

3.  **Install Dependencies**:
    Once your virtual environment is active, install the required project dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Data Acquisition

You have two options for obtaining the dataset:

  * **Option 1: Kaggle API (Recommended)**
    Ensure your Kaggle API key is configured correctly. Refer to the [Kaggle API documentation](https://www.kaggle.com/docs/api) for setup instructions.

    ```bash
    kaggle datasets download -d carrie1/ecommerce-data -p data/raw/ --unzip
    ```

  * **Option 2: Manual Download**

    1.  Go to the [UCI Online Retail Dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data) on Kaggle.
    2.  Download the `Online Retail.xlsx` file.
    3.  Extract the contents and place `data.csv` into the `data/raw/` directory within your project.

### Usage Instructions

Once the environment is set up and the data is acquired, you can execute the analytics pipeline and launch the dashboard.

1.  **Run Data Cleaning Pipeline**:

    ```bash
    python src/data_cleaning.py
    ```

    This script processes the raw data, handling missing values and outliers. It generates `online_retail_cleaned.csv` and a `cleaning_report.json` in `data/processed/`.

2.  **Run Exploratory Data Analysis (EDA)**:

    ```bash
    python src/eda_analysis.py
    ```

    This script performs comprehensive exploratory data analysis, generating insights on sales trends, customer behavior, and product performance. Note that this script primarily outputs insights to the console or internal memory and does not generate new persistent files.

3.  **Run Advanced Business Analytics**:

    ```bash
    python src/business_insights.py
    ```

    This script conducts deeper analyses, including RFM segmentation, cohort retention, market basket analysis, and customer lifetime value (CLV) modeling. Similar to the EDA script, its primary output is analytical insights.

4.  **Launch the Streamlit Dashboard**:

    ```bash
    streamlit run src/dashboard.py
    ```

    This command will open the interactive dashboard in your default web browser, typically at `http://localhost:8501`.

-----

## Project Structure

```
day-02-ecommerce-dashboard/
├── data/
│   ├── raw/                # Original raw dataset (e.g., Online Retail.xlsx or data.csv)
│   └── processed/          # Cleaned and processed datasets (e.g., online_retail_cleaned.csv)
├── src/
│   ├── data_cleaning.py    # Script for data preprocessing and cleaning
│   ├── eda_analysis.py     # Script for exploratory data analysis
│   ├── business_insights.py # Script for advanced business analytics and modeling
│   └── dashboard.py        # Streamlit application for the interactive dashboard
├── requirements.txt        # List of Python dependencies for the project
├── README.md               # This README file, providing project documentation
└── .gitignore              # Specifies intentionally untracked files to be ignored by Git
```

-----

## Dashboard Features

The interactive Streamlit dashboard provides a comprehensive and dynamic view of e-commerce performance.

### Executive Overview

  * **KPI Cards**: At-a-glance key performance indicators for total revenue, number of unique customers, products sold, and contributing countries, offering an immediate snapshot of business health.
  * **Trend Analysis**: Visualizes sales performance over time, allowing for the identification of seasonal trends, growth patterns, and anomalies.
  * **Geographic Distribution**: Provides a breakdown of revenue by country, highlighting top-performing regions and potential markets for expansion.

### Customer Analytics

  * **RFM Segmentation**: Interactive analysis of customer segments based on Recency, Frequency, and Monetary value, enabling highly targeted marketing campaigns and customer retention strategies.
  * **Cohort Analysis**: Heatmaps illustrating customer retention rates over time, revealing loyalty patterns and identifying periods of churn.
  * **CLV Distribution**: Insights into Customer Lifetime Value, helping to identify and nurture high-value customer segments.

### Product Intelligence

  * **Performance Metrics**: Identifies top-selling products and categories, guiding inventory management, merchandising, and marketing efforts.
  * **Market Basket Analysis**: Discovers strong product association rules (e.g., "customers who bought X also bought Y"), facilitating effective cross-selling and up-selling opportunities.
  * **Seasonal Patterns**: Analyzes sales performance based on seasonality, optimizing promotional timing and inventory levels.

### Advanced Filtering

The dashboard includes dynamic filters for granular data exploration:

  * **Date Range**: Analyze performance within specific timeframes.
  * **Country/Region**: Focus on sales data from particular geographic locations.
  * **Customer Segment**: Drill down into the behavior and characteristics of specific customer groups.
  * **Product Category**: Examine the performance of different product lines.

-----

## Technical Implementation

This section details the core components and their functionalities, providing insight into the project's technical architecture.

### Data Processing Pipeline (`data_cleaning.py`)

This script is responsible for ensuring data quality and preparing the raw dataset for subsequent analysis. Its key steps include:

1.  **Data Quality Assessment**: Initial checks for data integrity, completeness, and consistency.
2.  **Missing Value Handling**: Strategies to impute or remove incomplete data points.
3.  **Outlier Detection and Removal**: Identification and mitigation of anomalous data points that could skew analysis.
4.  **Feature Engineering**: Creation of new features (e.g., total price, invoice month) to enhance analytical capabilities.
5.  **Data Validation**: Ensures the cleaned data adheres to predefined rules and formats.
6.  **Export of the Cleaned Dataset**: Saves the processed data to `data/processed/` for use by other modules.

### Analytics Engine (`business_insights.py`)

This script houses the core business analytics logic, designed to extract deep insights from the cleaned data:

1.  **RFM Customer Segmentation**: Groups customers based on their Recency (last purchase), Frequency (how often they buy), and Monetary (how much they spend) values.
2.  **Cohort Retention Analysis**: Measures how well customers are retained over successive periods, often visualized as a heatmap.
3.  **Market Basket Analysis**: Utilizes algorithms like Apriori to uncover strong associations between purchased items.
4.  **Customer Lifetime Value (CLV) Calculation**: Estimates the total revenue a customer is expected to generate throughout their relationship with the business.
5.  **Time Series Forecasting**: Applies statistical models to predict future sales trends based on historical data.

### Dashboard Architecture (`dashboard.py`)

The Streamlit dashboard is built for interactivity and efficiency:

1.  **Efficient Data Loading and Caching**: Optimizes performance by loading and caching data, preventing redundant computations.
2.  **Interactive Sidebar Filters**: Provides intuitive controls for users to filter and explore data dynamically.
3.  **Dynamic Visualization Updates**: Ensures charts and graphs react instantly to user interactions and filter changes.
4.  **Real-time Metric Calculations**: Displays up-to-date Key Performance Indicators (KPIs) and analytical metrics.
5.  **Export Functionality for Reports**: Allows users to download selected data or visualizations for external reporting.

-----

## Business Impact

This project is designed to provide actionable insights that directly drive business growth and inform strategic decision-making.

### Customer Insights

  * **Champions (19.4%)**: These are your most valuable customers. They are ideal candidates for VIP treatment programs, exclusive offers, and loyalty initiatives to foster continued engagement.
  * **At Risk (1.4%)**: This small but critical segment requires immediate attention. Targeted retention campaigns, personalized outreach, or win-back strategies should be deployed to prevent churn.
  * **New Customers (5.7%)**: This segment represents a significant opportunity. Optimize onboarding processes and initial engagement strategies to convert them into loyal, repeat buyers.

### Revenue Optimization

  * **Product Focus**: The analysis shows that the **top 20% of products generate 76.2% of total revenue**. This critical insight should guide product development, marketing efforts, and inventory management, ensuring resources are allocated efficiently.
  * **Seasonal Planning**: A significant sales surge is observed in November. This necessitates proactive inventory stocking, staffing adjustments, and targeted promotional campaigns to fully capitalize on peak demand.
  * **International Expansion**: Analysis across 38 countries highlights diverse market performances. This data can inform targeted expansion strategies, identifying high-potential regions for localized marketing and distribution efforts.

### Strategic Recommendations

1.  **Retention Improvement**: Develop proactive strategies to enhance the initial **19.7%** first-period retention rate. Focus on improving the new customer experience to convert more one-time buyers into loyal customers.
2.  **Cross-selling Strategy**: Utilize insights from market basket analysis, particularly product association rules with high lift values (e.g., `lift > 6.0`), to implement effective cross-selling and up-selling campaigns. This can significantly increase the average order value.
3.  **Customer Segmentation**: Tailor marketing and communication efforts based on distinct RFM segments for higher engagement and conversion rates. Personalized offers and content will resonate more strongly with specific customer groups.
4.  **Inventory Optimization**: Leverage seasonal demand forecasts to optimize inventory levels. This will help minimize stockouts during peak seasons and reduce carrying costs associated with overstocking during quieter periods.

-----

## Dependencies

The project relies on the following Python libraries, specified with their exact versions to ensure reproducibility.

```txt
pandas==2.1.3
numpy==1.24.3
matplotlib==3.8.2
seaborn==0.12.2
plotly==5.17.0
streamlit==1.28.1
scikit-learn==1.3.2
kaggle==1.5.16
wordcloud==1.9.2
scipy==1.11.4
statsmodels==0.14.0
mlxtend>=0.21.0 # Note: Minimum version specified for broader compatibility.
```

-----