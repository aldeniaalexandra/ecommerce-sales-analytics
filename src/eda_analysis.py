import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EcommerceEDA:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        # Ensure proper data types
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def sales_trend_analysis(self):
        """Analyze sales trends over time"""
        print(" SALES TREND ANALYSIS")
        print("="*50)
        
        # Monthly sales trend
        monthly_sales = self.df.groupby(['Year', 'Month']).agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        }).rename(columns={
            'InvoiceNo': 'Transactions',
            'CustomerID': 'UniqueCustomers'
        })
        
        # Create date column for plotting
        monthly_sales.reset_index(inplace=True)
        monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
        
        # Plotly interactive chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Monthly Revenue', 'Monthly Transactions', 
                          'Monthly Unique Customers', 'Revenue vs Transactions'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Monthly Revenue
        fig.add_trace(
            go.Scatter(x=monthly_sales['Date'], y=monthly_sales['TotalAmount'],
                      mode='lines+markers', name='Revenue',
                      line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        # Monthly Transactions
        fig.add_trace(
            go.Scatter(x=monthly_sales['Date'], y=monthly_sales['Transactions'],
                      mode='lines+markers', name='Transactions',
                      line=dict(color='#ff7f0e', width=3)),
            row=1, col=2
        )
        
        # Monthly Customers
        fig.add_trace(
            go.Scatter(x=monthly_sales['Date'], y=monthly_sales['UniqueCustomers'],
                      mode='lines+markers', name='Customers',
                      line=dict(color='#2ca02c', width=3)),
            row=2, col=1
        )
        
        # Revenue vs Transactions scatter
        fig.add_trace(
            go.Scatter(x=monthly_sales['Transactions'], y=monthly_sales['TotalAmount'],
                      mode='markers', name='Revenue vs Trans',
                      marker=dict(size=10, color='#d62728')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Sales Trends Analysis Dashboard")
        fig.show()
        
        # Print insights
        total_revenue = monthly_sales['TotalAmount'].sum()
        avg_monthly_revenue = monthly_sales['TotalAmount'].mean()
        peak_month = monthly_sales.loc[monthly_sales['TotalAmount'].idxmax()]
        
        print(f" Key Insights:")
        print(f"  Total Revenue: £{total_revenue:,.2f}")
        print(f"  Average Monthly Revenue: £{avg_monthly_revenue:,.2f}")
        print(f"  Peak Month: {peak_month['Year']}-{peak_month['Month']:02d} (£{peak_month['TotalAmount']:,.2f})")
        
        return monthly_sales
    
    def customer_segmentation_analysis(self):
        """RFM Analysis and Customer Segmentation"""
        print("\n CUSTOMER SEGMENTATION (RFM ANALYSIS)")
        print("="*50)
        
        # Filter customers with CustomerID
        customer_data = self.df[self.df['HasCustomerID'] == True].copy()
        
        # Calculate RFM metrics
        snapshot_date = customer_data['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        rfm = customer_data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalAmount': 'sum'  # Monetary
        }).rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'TotalAmount': 'Monetary'
        })
        
        # Calculate RFM scores
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Calculate overall RFM Score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Customer Segmentation
        def segment_customers(row):
            if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['RFM_Score'] in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
                return 'Potential Loyalists'
            elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'New Customers'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['RFM_Score'] in ['245', '155', '154', '144', '135', '125', '124']:
                return 'Cannot Lose Them'
            else:
                return 'Others'
        
        rfm['Segment'] = rfm.apply(segment_customers, axis=1)
        
        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Customer Segments Distribution', 'RFM Scores Distribution',
                          'Monetary vs Frequency', 'Recency Distribution'],
            specs=[[{"type": "domain"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Customer segments pie chart
        segment_counts = rfm['Segment'].value_counts()
        fig.add_trace(
            go.Pie(labels=segment_counts.index, values=segment_counts.values,
                   name="Segments"),
            row=1, col=1
        )
        
        # RFM Score distribution
        fig.add_trace(
            go.Histogram(x=rfm['R_Score'], name='Recency Score', opacity=0.7),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=rfm['F_Score'], name='Frequency Score', opacity=0.7),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=rfm['M_Score'], name='Monetary Score', opacity=0.7),
            row=1, col=2
        )
        
        # Monetary vs Frequency scatter
        fig.add_trace(
            go.Scatter(x=rfm['Frequency'], y=rfm['Monetary'],
                      mode='markers', name='Customers',
                      marker=dict(size=5, opacity=0.6)),
            row=2, col=1
        )
        
        # Recency distribution
        fig.add_trace(
            go.Histogram(x=rfm['Recency'], name='Recency Days'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Customer Segmentation Analysis")
        fig.show()
        
        # Print segment insights
        print("\n Customer Segment Insights:")
        for segment in rfm['Segment'].unique():
            segment_data = rfm[rfm['Segment'] == segment]
            print(f"\n{segment}:")
            print(f"  Count: {len(segment_data):,} ({len(segment_data)/len(rfm)*100:.1f}%)")
            print(f"  Avg Recency: {segment_data['Recency'].mean():.1f} days")
            print(f"  Avg Frequency: {segment_data['Frequency'].mean():.1f} orders")
            print(f"  Avg Monetary: £{segment_data['Monetary'].mean():.2f}")
        
        return rfm
    
    def product_analysis(self):
        """Comprehensive Product Performance Analysis"""
        print("\n PRODUCT PERFORMANCE ANALYSIS")
        print("="*50)
        
        # Product performance metrics
        product_metrics = self.df.groupby(['StockCode', 'Description']).agg({
            'Quantity': 'sum',
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        }).rename(columns={
            'InvoiceNo': 'TransactionCount',
            'CustomerID': 'UniqueCustomers'
        })
        
        product_metrics['AvgOrderValue'] = product_metrics['TotalAmount'] / product_metrics['TransactionCount']
        product_metrics['AvgQuantityPerOrder'] = product_metrics['Quantity'] / product_metrics['TransactionCount']
        
        # Top products analysis
        top_products_by_revenue = product_metrics.nlargest(20, 'TotalAmount')
        top_products_by_quantity = product_metrics.nlargest(20, 'Quantity')
        
        # Category analysis
        category_performance = self.df.groupby('ProductCategory').agg({
            'TotalAmount': 'sum',
            'Quantity': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        }).rename(columns={
            'InvoiceNo': 'Transactions',
            'CustomerID': 'UniqueCustomers'
        })
        
        category_performance['AvgOrderValue'] = category_performance['TotalAmount'] / category_performance['Transactions']
        category_performance = category_performance.sort_values('TotalAmount', ascending=False)
        
        # Visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Top Products by Revenue', 'Category Performance',
                          'Product Popularity Matrix', 'Revenue Distribution'],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Top products by revenue
        top_10_revenue = top_products_by_revenue.head(10)
        fig.add_trace(
            go.Bar(x=top_10_revenue['TotalAmount'], 
                   y=[desc[:30] + '...' if len(desc) > 30 else desc 
                      for desc in top_10_revenue.index.get_level_values('Description')],
                   orientation='h', name='Revenue'),
            row=1, col=1
        )
        
        # Category performance
        fig.add_trace(
            go.Bar(x=category_performance.index, 
                   y=category_performance['TotalAmount'],
                   name='Category Revenue'),
            row=1, col=2
        )
        
        # Product popularity matrix (Revenue vs Quantity)
        sample_products = product_metrics.sample(min(500, len(product_metrics)))
        fig.add_trace(
            go.Scatter(x=sample_products['Quantity'], 
                      y=sample_products['TotalAmount'],
                      mode='markers', name='Products',
                      marker=dict(size=5, opacity=0.6)),
            row=2, col=1
        )
        
        # Revenue distribution
        fig.add_trace(
            go.Histogram(x=product_metrics['TotalAmount'], 
                        nbinsx=50, name='Revenue Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="Product Performance Analysis")
        fig.show()
        
        # Product insights
        total_products = len(product_metrics)
        top_20_percent_revenue = product_metrics.nlargest(int(total_products * 0.2), 'TotalAmount')['TotalAmount'].sum()
        total_revenue = product_metrics['TotalAmount'].sum()
        
        print(f"\n Product Insights:")
        print(f"  Total Unique Products: {total_products:,}")
        print(f"  Top 20% Products Revenue Share: {(top_20_percent_revenue/total_revenue*100):.1f}%")
        print(f"  Most Popular Category: {category_performance.index[0]}")
        print(f"  Highest Revenue Product: {top_products_by_revenue.index[0][1][:50]}...")
        
        return product_metrics, category_performance
    
    def seasonal_analysis(self):
        """Seasonal and Time-based Analysis"""
        print("\n SEASONAL & TIME-BASED ANALYSIS")
        print("="*50)
        
        # Seasonal performance
        seasonal_data = self.df.groupby('Season').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        })
        
        # Day of week analysis
        dow_data = self.df.groupby('DayName').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique'
        })
        
        # Reorder days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_data = dow_data.reindex(day_order)
        
        # Hourly analysis
        hourly_data = self.df.groupby('Hour').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique'
        })
        
        # Business period analysis
        period_data = self.df.groupby('BusinessPeriod').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique'
        })
        
        # Visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Seasonal Performance', 'Day of Week Analysis',
                          'Hourly Sales Pattern', 'Business Period Performance'],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Seasonal performance
        fig.add_trace(
            go.Bar(x=seasonal_data.index, y=seasonal_data['TotalAmount'],
                   name='Seasonal Revenue'),
            row=1, col=1
        )
        
        # Day of week
        fig.add_trace(
            go.Bar(x=dow_data.index, y=dow_data['TotalAmount'],
                   name='Daily Revenue'),
            row=1, col=2
        )
        
        # Hourly pattern
        fig.add_trace(
            go.Scatter(x=hourly_data.index, y=hourly_data['TotalAmount'],
                      mode='lines+markers', name='Hourly Revenue'),
            row=2, col=1
        )
        
        # Business periods
        fig.add_trace(
            go.Bar(x=period_data.index, y=period_data['TotalAmount'],
                   name='Period Revenue'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="Temporal Analysis Dashboard")
        fig.show()
        
        print(f"\n Temporal Insights:")
        print(f"  Best Season: {seasonal_data['TotalAmount'].idxmax()}")
        print(f"  Best Day: {dow_data['TotalAmount'].idxmax()}")
        print(f"  Peak Hour: {hourly_data['TotalAmount'].idxmax()}:00")
        print(f"  Best Period: {period_data['TotalAmount'].idxmax()}")
        
        return seasonal_data, dow_data, hourly_data
    
    def geographic_analysis(self):
        """Geographic Market Analysis"""
        print("\n GEOGRAPHIC MARKET ANALYSIS")
        print("="*50)
        
        # Country performance
        country_metrics = self.df.groupby('Country').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique',
            'Quantity': 'sum'
        }).rename(columns={
            'InvoiceNo': 'Transactions',
            'CustomerID': 'UniqueCustomers'
        })
        
        country_metrics['AvgOrderValue'] = country_metrics['TotalAmount'] / country_metrics['Transactions']
        country_metrics['RevenuePerCustomer'] = country_metrics['TotalAmount'] / country_metrics['UniqueCustomers']
        country_metrics = country_metrics.sort_values('TotalAmount', ascending=False)
        
        # Top 15 countries
        top_countries = country_metrics.head(15)
        
        # Visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Revenue by Country (Top 15)', 'Customers by Country',
                          'Average Order Value by Country', 'Market Share'],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "domain"}]]
        )
        
        # Revenue by country
        fig.add_trace(
            go.Bar(x=top_countries['TotalAmount'], y=top_countries.index,
                   orientation='h', name='Revenue'),
            row=1, col=1
        )
        
        # Customers by country
        fig.add_trace(
            go.Bar(x=top_countries['UniqueCustomers'], y=top_countries.index,
                   orientation='h', name='Customers'),
            row=1, col=2
        )
        
        # AOV by country
        fig.add_trace(
            go.Bar(x=top_countries['AvgOrderValue'], y=top_countries.index,
                   orientation='h', name='AOV'),
            row=2, col=1
        )
        
        # Market share pie
        fig.add_trace(
            go.Pie(labels=top_countries.index, values=top_countries['TotalAmount'],
                   name="Market Share"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="Geographic Analysis Dashboard")
        fig.show()
        
        # Geographic insights
        uk_share = (country_metrics.loc['United Kingdom', 'TotalAmount'] / 
                   country_metrics['TotalAmount'].sum() * 100)
        
        print(f"\n Geographic Insights:")
        print(f"  Total Countries: {len(country_metrics)}")
        print(f"  UK Market Share: {uk_share:.1f}%")
        print(f"  Highest AOV Country: {country_metrics['AvgOrderValue'].idxmax()}")
        print(f"  Most Customers: {country_metrics['UniqueCustomers'].idxmax()}")
        
        return country_metrics
    
    def generate_comprehensive_report(self):
        """Generate comprehensive EDA report"""
        print("\n" + "="*60)
        print(" COMPREHENSIVE EDA REPORT")
        print("="*60)
        
        # Execute all analyses
        monthly_trends = self.sales_trend_analysis()
        rfm_analysis = self.customer_segmentation_analysis()
        product_metrics, category_performance = self.product_analysis()
        seasonal_data, dow_data, hourly_data = self.seasonal_analysis()
        country_metrics = self.geographic_analysis()
        
        # Overall business metrics
        total_revenue = self.df['TotalAmount'].sum()
        total_transactions = self.df['InvoiceNo'].nunique()
        total_customers = self.df['CustomerID'].nunique()
        total_products = self.df['StockCode'].nunique()
        
        print(f"\n EXECUTIVE SUMMARY:")
        print(f"  Total Revenue: £{total_revenue:,.2f}")
        print(f"  Total Transactions: {total_transactions:,}")
        print(f"  Total Customers: {total_customers:,}")
        print(f"  Total Products: {total_products:,}")
        print(f"  Average Order Value: £{total_revenue/total_transactions:.2f}")
        print(f"  Revenue per Customer: £{total_revenue/total_customers:.2f}")
        
        return {
            'monthly_trends': monthly_trends,
            'rfm_analysis': rfm_analysis,
            'product_metrics': product_metrics,
            'category_performance': category_performance,
            'seasonal_data': seasonal_data,
            'country_metrics': country_metrics
        }

# Usage Example
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / "data" / "processed" / "online_retail_cleaned.csv"

    eda = EcommerceEDA(input_path)
    comprehensive_report = eda.generate_comprehensive_report()
