import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
# For Market Basket Analysis
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="E-commerce Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        color: #333333;
    }
    .insight-box h4 {
        color: #2E86AB !important;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6 !important;
        color: #333333 !important;
    }
    .stSelectbox div[role="listbox"] > div {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the cleaned dataset and calculate RFM."""
    try:
        df = pd.read_csv('data/processed/online_retail_cleaned.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # Calculate RFM scores
        if 'HasCustomerID' in df.columns:
            customer_data = df[df['HasCustomerID'] == True].copy()
        else:
            customer_data = df.dropna(subset=['CustomerID']).copy()
        
        if not customer_data.empty:
            snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
            rfm = customer_data.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
                'InvoiceNo': 'nunique',
                'TotalAmount': 'sum'
            }).rename(columns={
                'InvoiceDate': 'Recency',
                'InvoiceNo': 'Frequency',
                'TotalAmount': 'Monetary'
            })
            
            # Create segments based on quantiles (using 5 quantiles for more granular segments)
            # Ensure unique bin edges for qcut
            for col in ['Recency', 'Frequency', 'Monetary']:
                if rfm[col].nunique() < 5: # If not enough unique values for 5 bins
                    rfm[col + '_Score'] = pd.qcut(rfm[col], q=rfm[col].nunique(), labels=False, duplicates='drop') + 1
                    rfm[col + '_Score'] = pd.cut(rfm[col], bins=rfm[col].nunique(), labels=False, include_lowest=True, duplicates='drop') + 1
                else:
                    rfm[col + '_Score'] = pd.qcut(rfm[col], 5, labels=False, duplicates='drop') + 1 # labels=[1,2,3,4,5]

            # Invert Recency score
            rfm['R_Score'] = rfm['Recency_Score'].astype(int).apply(lambda x: 6 - x if x is not None else None)
            rfm['F_Score'] = rfm['Frequency_Score'].astype(int)
            rfm['M_Score'] = rfm['Monetary_Score'].astype(int)
            
            # Combine RFM scores into a segment
            rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

            # Define specific segment names
            def rfm_segment_name(row):
                r = row['R_Score']
                f = row['F_Score']
                m = row['M_Score']

                if r >= 4 and f >= 4 and m >= 4:
                    return 'Champions'
                elif r >= 4 and f >= 3 and m >= 3:
                    return 'Loyal Customers'
                elif r >= 4 and f >= 1 and m >= 1:
                    return 'New Customers'
                elif r <= 2 and f >= 3 and m >= 3:
                    return 'At Risk'
                elif r <= 2 and f <= 2 and m <= 2:
                    return 'Churned/Lost'
                elif r >= 3 and f >= 2 and m >= 2:
                    return 'Potentially Loyal'
                else:
                    return 'Others'

            rfm['Customer_Segment'] = rfm.apply(rfm_segment_name, axis=1)

            # Merge RFM data back to the original DataFrame
            # Make sure CustomerID in df is also int/str consistent
            df['CustomerID'] = df['CustomerID'].fillna(0).astype(int)
            rfm.index = rfm.index.fillna(0).astype(int)

            df = df.merge(rfm[['RFM_Segment', 'Customer_Segment']], on='CustomerID', how='left')
        else:
            df['RFM_Segment'] = np.nan
            df['Customer_Segment'] = np.nan

        return df
    except FileNotFoundError:
        st.error("❌ Cleaned dataset not found. Please run the data cleaning pipeline first.")
        st.stop()

@st.cache_data
def calculate_key_metrics(df):
    """Calculate and cache key business metrics using actual column names"""
    total_revenue = df['TotalAmount'].sum()
    total_transactions = df['InvoiceNo'].nunique()
    
    if 'HasCustomerID' in df.columns:
        total_customers = df[df['HasCustomerID'] == True]['CustomerID'].nunique()
        customer_transactions = len(df[df['HasCustomerID'] == True].dropna(subset=['CustomerID'])) # Only count transactions with CustomerID
    else:
        total_customers = df.dropna(subset=['CustomerID'])['CustomerID'].nunique()
        customer_transactions = len(df.dropna(subset=['CustomerID']))
    
    total_products = df['StockCode'].nunique()
    avg_order_value = total_revenue / total_transactions if total_transactions > 0 else 0
    revenue_per_customer = total_revenue / total_customers if total_customers > 0 else 0
    
    return {
        'total_revenue': total_revenue,
        'total_transactions': total_transactions,
        'total_customers': total_customers,
        'total_products': total_products,
        'avg_order_value': avg_order_value,
        'revenue_per_customer': revenue_per_customer,
        'customer_transactions': customer_transactions # Number of transactions from identified customers
    }


def create_time_series_chart(df, date_col, value_col, title):
    """Create interactive time series chart"""
    daily_data = df.groupby(df[date_col].dt.date)[value_col].sum().reset_index()
    daily_data.columns = ['Date', 'Value']
    
    fig = px.line(daily_data, x='Date', y='Value', title=title)
    fig.update_traces(line_color='#2E86AB', line_width=3)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        title_font_size=16
    )
    return fig

def create_top_products_chart(df, top_n=10):
    """Create top products by revenue chart"""
    product_revenue = df.groupby('Description')['TotalAmount'].sum().nlargest(top_n)
    
    fig = px.bar(
        x=product_revenue.values,
        y=[desc[:40] + '...' if len(desc) > 40 else desc for desc in product_revenue.index],
        orientation='h',
        title=f'Top {top_n} Products by Revenue'
    )
    fig.update_traces(marker_color='#764ba2')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_customer_segmentation_chart(df):
    """Create customer segmentation visualization"""
    if 'HasCustomerID' in df.columns:
        customer_data = df[df['HasCustomerID'] == True]
    else:
        customer_data = df.dropna(subset=['CustomerID'])
    
    if customer_data.empty:
        return None, None
    
    # Use the pre-calculated Customer_Segment from load_data
    if 'Customer_Segment' not in customer_data.columns:
        st.warning("Customer Segmentation data not found. Please ensure 'Customer_Segment' column is created in data loading.")
        return None, None

    segment_counts = customer_data.groupby('Customer_Segment')['CustomerID'].nunique().reset_index()
    segment_counts.columns = ['Customer_Segment', 'Count']
    segment_counts['Percentage'] = (segment_counts['Count'] / segment_counts['Count'].sum()) * 100
    segment_counts = segment_counts.sort_values('Percentage', ascending=False)
    
    fig = px.bar(segment_counts, x='Customer_Segment', y='Percentage',
                 title='Customer Segments Distribution',
                 text_auto='.1f%',
                 color='Percentage',
                 color_continuous_scale=px.colors.sequential.Tealgrn) # Changed color scale
    fig.update_traces(marker_color='#667eea')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        xaxis_tickangle=-45,
        yaxis_title="Percentage of Customers"
    )
    return fig, segment_counts # Return both figure and data

def create_seasonal_analysis_chart(df):
    """Create seasonal analysis based on actual data"""
    if 'Season' in df.columns:
        seasonal_data = df.groupby('Season').agg({
            'TotalAmount': ['sum', 'mean'],
            'InvoiceNo': 'nunique'
        }).round(2)
        
        seasonal_data.columns = ['Total_Revenue', 'Avg_Revenue', 'Transaction_Count']
        seasonal_data = seasonal_data.reset_index()
        
        fig = px.bar(seasonal_data, x='Season', y='Total_Revenue',
                     title='Revenue Distribution by Season',
                     color='Total_Revenue', color_continuous_scale='Blues')
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig, seasonal_data
    return None, None

def create_hourly_pattern_chart(df):
    """Create hourly sales pattern chart"""
    if 'Hour' in df.columns:
        hourly_data = df.groupby('Hour').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hourly_data['Hour'],
            y=hourly_data['TotalAmount'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#2E86AB', width=3),
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_data['Hour'],
            y=hourly_data['InvoiceNo'],
            mode='lines+markers',
            name='Transactions',
            line=dict(color='#764ba2', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Sales Pattern by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis=dict(title='Revenue (£)', side='left'),
            yaxis2=dict(title='Transaction Count', side='right', overlaying='y'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig, hourly_data
    return None, None

def create_geographic_analysis(df):
    """Create geographic analysis with proper handling of country data"""
    country_data = df.groupby('Country').agg({
        'TotalAmount': ['sum', 'mean'],
        'InvoiceNo': 'nunique',
        'CustomerID': lambda x: x[df.loc[x.index, 'HasCustomerID'] == True].nunique() if 'HasCustomerID' in df.columns else x.nunique()
    }).round(2)
    
    country_data.columns = ['Total_Revenue', 'Avg_Order_Value', 'Transaction_Count', 'Customer_Count']
    country_data = country_data.reset_index()
    country_data = country_data.sort_values('Total_Revenue', ascending=False)
    
    # Top 10 countries by revenue
    top_countries = country_data.head(10)
    
    fig = px.bar(top_countries, x='Total_Revenue', y='Country',
                 orientation='h', title='Top 10 Countries by Revenue',
                 color='Total_Revenue', color_continuous_scale='Blues')
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig, country_data

def create_cohort_analysis_chart(df):
    """Create a cohort analysis heatmap."""
    if 'HasCustomerID' in df.columns:
        cohort_df = df[df['HasCustomerID'] == True].copy()
    else:
        cohort_df = df.dropna(subset=['CustomerID']).copy()

    if cohort_df.empty:
        return None, None

    cohort_df['InvoiceMonth'] = cohort_df['InvoiceDate'].dt.to_period('M')
    cohort_df['AcquisitionMonth'] = cohort_df.groupby('CustomerID')['InvoiceMonth'].transform('min')

    cohort_df['CohortPeriod'] = (cohort_df['InvoiceMonth'] - cohort_df['AcquisitionMonth']).apply(lambda x: x.n)

    cohort_counts = cohort_df.groupby(['AcquisitionMonth', 'CohortPeriod'])['CustomerID'].nunique().reset_index()

    cohort_pivot = cohort_counts.pivot_table(index='AcquisitionMonth',
                                             columns='CohortPeriod',
                                             values='CustomerID')

    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)

    retention_matrix.index = retention_matrix.index.astype(str)
    retention_matrix.columns = retention_matrix.columns.astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=retention_matrix.values,
        x=retention_matrix.columns,
        y=retention_matrix.index,
        colorscale='Blues',
        colorbar_title='Retention Rate',
        texttemplate="%{z:.0%}",
        textfont={"size":10}
    ))

    fig.update_layout(
        title='Customer Retention Cohort Analysis',
        xaxis_title='Cohort Period (Months)',
        yaxis_title='Acquisition Month',
        xaxis_nticks=len(retention_matrix.columns),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333')
    )
    return fig, retention_matrix

def perform_market_basket_analysis(df, min_support=0.01, min_confidence=0.5):
    """Perform Market Basket Analysis and return association rules."""
    if 'HasCustomerID' in df.columns:
        mba_df = df[df['HasCustomerID'] == True].copy()
    else:
        mba_df = df.dropna(subset=['CustomerID']).copy()

    if mba_df.empty:
        return pd.DataFrame() # Return empty DataFrame if no customer data

    # Aggregate items by InvoiceNo to form transactions
    transactions = mba_df.groupby('InvoiceNo')['Description'].apply(list).values

    # Encode transactions
    te = TransactionEncoder()
    try:
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    except ValueError as e:
        st.warning(f"Error during transaction encoding: {e}. Skipping Market Basket Analysis.")
        return pd.DataFrame()

    # Apply Apriori algorithm
    try:
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    except Exception as e:
        st.warning(f"Error during Apriori algorithm: {e}. Skipping Market Basket Analysis.")
        return pd.DataFrame()
    
    if frequent_itemsets.empty:
        return pd.DataFrame()

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_confidence)
    rules = rules.sort_values(['lift'], ascending=False).reset_index(drop=True)
    
    if not rules.empty:
        # Convert frozensets to strings for better display
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    return rules


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 E-commerce Business Intelligence Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    metrics = calculate_key_metrics(df)
    
    # Sidebar filters
    st.sidebar.header("Dashboard Filters")
    
    # Date range filter
    min_date = df['InvoiceDate'].min().date()
    max_date = df['InvoiceDate'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Country filter
    countries = sorted(df['Country'].unique().tolist())
    selected_countries = st.sidebar.multiselect("Select Countries", countries)
    
    # Product category filter
    if 'ProductCategory' in df.columns and not df['ProductCategory'].isnull().all():
        categories = sorted(df['ProductCategory'].dropna().unique().tolist())
        selected_categories = st.sidebar.multiselect("Select Product Categories", categories)
    else:
        selected_categories = []

    # Customer Segment filter
    if 'Customer_Segment' in df.columns and not df['Customer_Segment'].isnull().all():
        customer_segments = ['All'] + sorted(df['Customer_Segment'].dropna().unique().tolist())
        selected_customer_segment = st.sidebar.selectbox("Filter by Customer Segment", customer_segments)
    else:
        selected_customer_segment = 'All' # Default if column not present


    # Apply filters
    filtered_df = df[
        (df['InvoiceDate'].dt.date >= date_range[0]) &
        (df['InvoiceDate'].dt.date <= date_range[1])
    ]

    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

    if selected_categories and 'ProductCategory' in df.columns:
        filtered_df = filtered_df[filtered_df['ProductCategory'].isin(selected_categories)]

    if selected_customer_segment != 'All' and 'Customer_Segment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Customer_Segment'] == selected_customer_segment]

    # Calculate filtered metrics
    filtered_metrics = calculate_key_metrics(filtered_df)
    
    # Key Metrics Dashboard
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        filtered_revenue = filtered_df['TotalAmount'].sum()
        revenue_change = ((filtered_revenue / metrics['total_revenue']) - 1) * 100 if metrics['total_revenue'] > 0 else 0
        st.metric(
            label="Total Revenue",
            value=f"£{filtered_revenue:,.0f}",
            delta=f"{revenue_change:+.1f}% of total"
        )
    
    with col2:
        filtered_transactions = filtered_df['InvoiceNo'].nunique()
        trans_change = filtered_transactions - metrics['total_transactions']
        st.metric(
            label="Transactions",
            value=f"{filtered_transactions:,}",
            delta=f"{trans_change:+,} vs total"
        )
    
    with col3:
        if 'HasCustomerID' in filtered_df.columns:
            filtered_customers = filtered_df[filtered_df['HasCustomerID'] == True]['CustomerID'].nunique()
        else:
            filtered_customers = filtered_df.dropna(subset=['CustomerID'])['CustomerID'].nunique()
        
        cust_change = filtered_customers - metrics['total_customers']
        st.metric(
            label="Unique Customers",
            value=f"{filtered_customers:,}",
            delta=f"{cust_change:+,} vs total"
        )
    
    with col4:
        filtered_aov = filtered_revenue / filtered_transactions if filtered_transactions > 0 else 0
        aov_change = ((filtered_aov / metrics['avg_order_value']) - 1) * 100 if metrics['avg_order_value'] > 0 else 0
        st.metric(
            label="Avg Order Value",
            value=f"£{filtered_aov:.2f}",
            delta=f"{aov_change:+.1f}% vs total"
        )
    
    with col5:
        filtered_products = filtered_df['StockCode'].nunique()
        prod_change = filtered_products - metrics['total_products']
        st.metric(
            label="Products Sold",
            value=f"{filtered_products:,}",
            delta=f"{prod_change:+,} vs total"
        )
    
    # Charts Section
    
    # Row 1: Time Series and Geographic Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue trend over time
        time_chart = create_time_series_chart(
            filtered_df, 'InvoiceDate', 'TotalAmount', 
            'Daily Revenue Trend'
        )
        st.plotly_chart(time_chart, use_container_width=True)
    
    with col2:
        # Top countries by revenue
        if len(filtered_df) > 0:
            country_revenue = filtered_df.groupby('Country')['TotalAmount'].sum().nlargest(10)
            if not country_revenue.empty:
                country_fig = px.pie(
                    values=country_revenue.values,
                    names=country_revenue.index,
                    title='Revenue Distribution by Country (Top 10)'
                )
                country_fig.update_traces(textposition='inside', textinfo='percent+label')
                country_fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(country_fig, use_container_width=True)
            else:
                st.info("No country data to display for the current filters.")
    
    # Row for Customer Analytics
    st.markdown("### Customer Analytics")
    col1, col2 = st.columns(2)

    with col1:
        # Customer Segmentation Chart
        if len(filtered_df) > 0:
            customer_segment_chart, _ = create_customer_segmentation_chart(filtered_df)
            if customer_segment_chart:
                st.plotly_chart(customer_segment_chart, use_container_width=True)
            else:
                st.info("No customer segmentation data to display for the current filters (requires CustomerID).")
        else:
            st.info("No data to display customer segmentation.")
    
    with col2:
        # Cohort Analysis
        if len(filtered_df) > 0:
            cohort_chart, _ = create_cohort_analysis_chart(filtered_df)
            if cohort_chart:
                st.plotly_chart(cohort_chart, use_container_width=True)
            else:
                st.info("No cohort analysis data to display for the current filters (requires CustomerID).")
        else:
            st.info("No data to display cohort analysis.")
    
    # Row 2: Product Analysis
    st.markdown("### Product Intelligence")
    col1, col2 = st.columns(2)
    
    with col1:
        # Top products chart
        if len(filtered_df) > 0:
            products_chart = create_top_products_chart(filtered_df, 15)
            st.plotly_chart(products_chart, use_container_width=True)
    
    with col2:
        # Product category performance
        if 'ProductCategory' in filtered_df.columns and not filtered_df['ProductCategory'].isnull().all():
            category_revenue = filtered_df.groupby('ProductCategory')['TotalAmount'].sum().sort_values(ascending=True)
            if not category_revenue.empty:
                category_fig = px.bar(
                    x=category_revenue.values,
                    y=category_revenue.index,
                    orientation='h',
                    title='Revenue by Product Category'
                )
                category_fig.update_traces(marker_color='#2E86AB')
                category_fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(category_fig, use_container_width=True)
            else:
                st.info("No product category data to display for the current filters.")
        else:
            st.info("No product category data available.")

    # Market Basket Analysis
    st.markdown("### Market Basket Analysis (Product Associations)")
    if st.checkbox("Show Product Association Rules"):
        st.info("Finding products often purchased together. This may take a moment.")
        mba_rules = perform_market_basket_analysis(filtered_df, min_support=0.01, min_confidence=0.5)
        if mba_rules is not None and not mba_rules.empty:
            st.dataframe(mba_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20), use_container_width=True)
            st.markdown("""
            <div class="insight-box">
            <h4>💡 Market Basket Insights</h4>
            <ul>
            <li><strong>Lift > 1:</strong> Indicates a positive association (products are purchased together more often than expected by chance).</li>
            <li><strong>Confidence:</strong> The probability that a customer buys the consequent given that they have already bought the antecedent.</li>
            <li><strong>Support:</strong> The popularity of an itemset in terms of its occurrences in the transactions.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No strong association rules found for the selected filters or not enough customer data. Try adjusting date range or country.")
    
    # Row 3:Temporal Analysis
    st.markdown("### Temporal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal analysis
        seasonal_chart, seasonal_data = create_seasonal_analysis_chart(filtered_df)
        if seasonal_chart:
            st.plotly_chart(seasonal_chart, use_container_width=True)
    
    with col2:
        # Hourly pattern
        hourly_chart, hourly_data = create_hourly_pattern_chart(filtered_df)
        if hourly_chart:
            st.plotly_chart(hourly_chart, use_container_width=True)
    
    # Row 4:Geographic Analysis
    st.markdown("### Geographic Market Analysis")
    geo_chart, geo_data = create_geographic_analysis(filtered_df)
    if geo_chart:
        st.plotly_chart(geo_chart, use_container_width=True)
        
        # Market insights
        if not geo_data.empty:
            uk_revenue = geo_data[geo_data['Country'] == 'United Kingdom']['Total_Revenue'].iloc[0] if 'United Kingdom' in geo_data['Country'].values else 0
            total_revenue = geo_data['Total_Revenue'].sum()
            uk_share = (uk_revenue / total_revenue * 100) if total_revenue > 0 else 0
            
            top_international_country = geo_data[geo_data['Country'] != 'United Kingdom'].iloc[0]['Country'] if len(geo_data[geo_data['Country'] != 'United Kingdom']) > 0 else 'N/A'

            st.markdown(f"""
            <div class="insight-box">
            <h4>🎯 Geographic Insights</h4>
            <ul>
            <li><strong>UK Market Dominance:</strong> {uk_share:.1f}% of total revenue</li>
            <li><strong>International Markets:</strong> {len(geo_data) - (1 if 'United Kingdom' in geo_data['Country'].values else 0)} countries contributing {100-uk_share:.1f}% of revenue</li>
            <li><strong>Top International Market:</strong> {top_international_country}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No geographic insights to display for the current filters.")
    
    # Row 5: Business Insights Section with Real Data
    st.markdown("## Data-Driven Business Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        # Calculate actual insights from data
        peak_hour = hourly_data.loc[hourly_data['TotalAmount'].idxmax(), 'Hour'] if hourly_data is not None and not hourly_data.empty else 'Unknown'
        peak_season = seasonal_data.loc[seasonal_data['Total_Revenue'].idxmax(), 'Season'] if seasonal_data is not None and not seasonal_data.empty else 'Unknown'
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>🎯 Key Performance Insights</h4>
        <ul>
        <li><strong>Peak Sales Hour:</strong> {peak_hour}:00 - Optimize staff and inventory during peak hours.</li>
        <li><strong>Best Season:</strong> {peak_season} - Plan seasonal campaigns and promotions.</li>
        <li><strong>Customer Retention:</strong> Focus on improving customer retention and converting guest customers to registered users.</li>
        <li><strong>AOV Optimization:</strong> Current AOV £{filtered_metrics['avg_order_value']:.2f} - There is potential for bundling offers or upselling.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        # Revenue insights
        avg_orders_per_customer = filtered_metrics['customer_transactions'] / filtered_metrics['total_customers'] if filtered_metrics['total_customers'] > 0 else 0
            
        st.markdown(f"""
        <div class="insight-box">
        <h4>📈 Strategic Recommendations</h4>
        <ul>
        <li><strong>Customer Lifetime Value:</strong> Average £{filtered_metrics['revenue_per_customer']:.2f} per customer - target high-value customers with exclusive offers.</li>
        <li><strong>Cross-selling Potential:</strong> Average {avg_orders_per_customer:.1f} transactions per registered customer - identify related products for cross-selling.</li>
        <li><strong>Market Expansion:</strong> Identify and leverage growing international markets based on geographical performance.</li>
        <li><strong>Inventory Strategy:</strong> Optimize stock based on product category performance and seasonal patterns.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Export Section
    st.markdown("## Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ecommerce_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Summary Report"):
            summary_data = {
                'Metric': ['Total Revenue', 'Total Transactions', 'Unique Customers', 'Avg Order Value', 'Products Sold'],
                'Value': [f"£{filtered_revenue:,.2f}", f"{filtered_transactions:,}", 
                         f"{filtered_customers:,}", f"£{filtered_aov:.2f}", f"{filtered_products:,}"]
            }
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary",
                data=csv,
                file_name=f"ecommerce_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("View Raw Data Sample"):
            st.dataframe(filtered_df.head(100), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666; font-size: 0.9em;'>
    E-commerce Business Intelligence Dashboard | Built with Streamlit & Plotly<br>
    Data Analysis for Strategic Decision Making
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()