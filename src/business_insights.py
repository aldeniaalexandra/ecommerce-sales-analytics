import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

class BusinessInsights:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        
    def cohort_analysis(self):
        """Customer Cohort Retention Analysis"""
        print(" COHORT RETENTION ANALYSIS")
        print("="*50)
        
        # Filter customers with CustomerID
        customer_data = self.df[self.df['HasCustomerID'] == True].copy()
        
        # Get customer's first purchase date
        customer_data['Period'] = customer_data['InvoiceDate'].dt.to_period('M')
        customer_data['CohortGroup'] = customer_data.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
        
        # Calculate period number
        def get_period_number(df):
            df['PeriodNumber'] = (df['Period'] - df['CohortGroup']).apply(attrgetter('n'))
            return df
        
        from operator import attrgetter
        customer_data = customer_data.groupby(level=0).apply(get_period_number).reset_index(drop=True)
        
        # Create cohort table
        cohort_data = customer_data.groupby(['CohortGroup', 'PeriodNumber'])['CustomerID'].nunique().reset_index()
        cohort_sizes = customer_data.groupby('CohortGroup')['CustomerID'].nunique().reset_index()
        cohort_table = cohort_data.merge(cohort_sizes, on='CohortGroup', suffixes=['', '_cohort_size'])
        cohort_table['RetentionRate'] = cohort_table['CustomerID'] / cohort_table['CustomerID_cohort_size']
        
        # Pivot for heatmap
        cohort_table_pivot = cohort_table.pivot_table(
            index='CohortGroup',
            columns='PeriodNumber', 
            values='RetentionRate'
        )
        
        # Visualization
        fig = go.Figure(data=go.Heatmap(
            z=cohort_table_pivot.values,
            x=[f'Period {i}' for i in cohort_table_pivot.columns],
            y=[str(idx) for idx in cohort_table_pivot.index],
            colorscale='Blues',
            text=np.round(cohort_table_pivot.values * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Customer Cohort Retention Analysis (%)',
            xaxis_title='Period Number',
            yaxis_title='Cohort Group',
            height=600
        )
        fig.show()
        
        # Calculate average retention rates
        avg_retention = cohort_table_pivot.mean(axis=0)
        print(f"\n Retention Insights:")
        for period, rate in avg_retention.items():
            if pd.notna(rate):
                print(f"  Period {period}: {rate*100:.1f}% retention")
        
        return cohort_table_pivot
    
    def market_basket_analysis(self):
        """Market Basket Analysis - Frequently Bought Together"""
        print("\n MARKET BASKET ANALYSIS")
        print("="*50)
        
        # Create transaction-product matrix
        basket = (self.df.groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().fillna(0))
        
        # Convert to binary (bought/not bought)
        basket_sets = basket.map(lambda x: 1 if x > 0 else 0)
        
        # Calculate support (frequency of itemsets)
        item_support = basket_sets.mean()
        popular_items = item_support[item_support >= 0.01].sort_values(ascending=False)
        
        # Calculate association rules for top items
        from itertools import combinations
        
        # Get top 20 items for association analysis
        top_items = popular_items.head(20).index.tolist()
        
        # Calculate lift for item pairs
        associations = []
        for item_a, item_b in combinations(top_items, 2):
            # Support of individual items
            support_a = basket_sets[item_a].mean()
            support_b = basket_sets[item_b].mean()
            
            # Support of itemset {A, B}
            support_ab = ((basket_sets[item_a] == 1) & (basket_sets[item_b] == 1)).mean()
            
            # Calculate confidence and lift
            if support_a > 0 and support_b > 0:
                confidence_a_to_b = support_ab / support_a if support_a > 0 else 0
                confidence_b_to_a = support_ab / support_b if support_b > 0 else 0
                lift = support_ab / (support_a * support_b) if (support_a * support_b) > 0 else 0
                
                if lift > 1.2 and support_ab >= 0.005:  # Only strong associations
                    associations.append({
                        'Item_A': item_a[:30],
                        'Item_B': item_b[:30],
                        'Support_AB': support_ab,
                        'Confidence_A_to_B': confidence_a_to_b,
                        'Confidence_B_to_A': confidence_b_to_a,
                        'Lift': lift
                    })
        
        # Convert to DataFrame and sort by lift
        association_df = pd.DataFrame(associations).sort_values('Lift', ascending=False)
        
        # Visualization
        if len(association_df) > 0:
            top_associations = association_df.head(15)
            
            fig = go.Figure(data=go.Scatter(
                x=top_associations['Support_AB'],
                y=top_associations['Confidence_A_to_B'],
                mode='markers+text',
                marker=dict(
                    size=top_associations['Lift'] * 10,
                    color=top_associations['Lift'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Lift")
                ),
                text=top_associations.apply(lambda x: f"{x['Item_A'][:15]} → {x['Item_B'][:15]}", axis=1),
                textposition="middle right",
                hovertemplate='<b>%{text}</b><br>' +
                             'Support: %{x:.3f}<br>' +
                             'Confidence: %{y:.3f}<br>' +
                             'Lift: %{marker.color:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Market Basket Analysis - Product Associations',
                xaxis_title='Support (A ∩ B)',
                yaxis_title='Confidence (A → B)',
                height=600,
                showlegend=False
            )
            fig.show()
            
            print(f"\n Top Product Associations:")
            for _, row in association_df.head(10).iterrows():
                print(f"  {row['Item_A']} → {row['Item_B']}")
                print(f"    Lift: {row['Lift']:.2f}, Confidence: {row['Confidence_A_to_B']:.1%}")
        
        return association_df if len(association_df) > 0 else None
    
    def customer_lifetime_value(self):
        """Customer Lifetime Value Analysis"""
        print("\n CUSTOMER LIFETIME VALUE ANALYSIS")
        print("="*50)
        
        # Filter customers with CustomerID
        customer_data = self.df[self.df['HasCustomerID'] == True].copy()
        
        # Calculate CLV components
        clv_data = customer_data.groupby('CustomerID').agg({
            'TotalAmount': ['sum', 'mean'],
            'InvoiceNo': 'nunique',
            'InvoiceDate': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        clv_data.columns = ['TotalRevenue', 'AvgOrderValue', 'Frequency', 'FirstPurchase', 'LastPurchase']
        
        # Calculate customer lifespan in days
        clv_data['LifespanDays'] = (clv_data['LastPurchase'] - clv_data['FirstPurchase']).dt.days
        clv_data['LifespanDays'] = clv_data['LifespanDays'].fillna(0)  # Single purchase customers
        
        # Calculate purchase frequency (orders per day)
        clv_data['PurchaseFrequency'] = clv_data['Frequency'] / (clv_data['LifespanDays'] + 1)
        
        # Simple CLV calculation: AOV * Frequency * Predicted Lifespan
        # For customers with single purchase, use average lifespan
        avg_lifespan = clv_data[clv_data['LifespanDays'] > 0]['LifespanDays'].mean()
        clv_data['PredictedLifespan'] = np.where(
            clv_data['LifespanDays'] == 0, 
            avg_lifespan, 
            clv_data['LifespanDays']
        )
        
        # Calculate CLV
        clv_data['CLV'] = (clv_data['AvgOrderValue'] * 
                          clv_data['PurchaseFrequency'] * 
                          clv_data['PredictedLifespan'])
        
        # Customer value segments
        clv_data['ValueSegment'] = pd.qcut(
            clv_data['CLV'], 
            q=5, 
            labels=['Low Value', 'Medium-Low', 'Medium', 'Medium-High', 'High Value']
        )
        
        # Visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['CLV Distribution', 'Value Segments',
                          'CLV vs Total Revenue', 'AOV vs Frequency'],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # CLV distribution
        fig.add_trace(
            go.Histogram(x=clv_data['CLV'], nbinsx=50, name='CLV Distribution'),
            row=1, col=1
        )
        
        # Value segments
        segment_counts = clv_data['ValueSegment'].value_counts()
        fig.add_trace(
            go.Bar(x=segment_counts.index, y=segment_counts.values, name='Segments'),
            row=1, col=2
        )
        
        # CLV vs Total Revenue
        sample_customers = clv_data.sample(min(1000, len(clv_data)))
        fig.add_trace(
            go.Scatter(
                x=sample_customers['TotalRevenue'], 
                y=sample_customers['CLV'],
                mode='markers', 
                name='Customers',
                marker=dict(size=4, opacity=0.6)
            ),
            row=2, col=1
        )
        
        # AOV vs Frequency
        fig.add_trace(
            go.Scatter(
                x=sample_customers['AvgOrderValue'], 
                y=sample_customers['Frequency'],
                mode='markers', 
                name='Customer Behavior',
                marker=dict(size=4, opacity=0.6)
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="Customer Lifetime Value Analysis")
        fig.show()
        
        # CLV insights
        print(f"\n CLV Insights:")
        print(f"  Average CLV: £{clv_data['CLV'].mean():.2f}")
        print(f"  Median CLV: £{clv_data['CLV'].median():.2f}")
        print(f"  Top 10% CLV: £{clv_data['CLV'].quantile(0.9):.2f}")
        print(f"  High Value Customers: {len(clv_data[clv_data['ValueSegment'] == 'High Value']):,}")
        
        return clv_data
    
    def seasonal_forecasting(self):
        """Basic Seasonal Sales Forecasting"""
        print("\n SEASONAL SALES FORECASTING")
        print("="*50)
        
        # Monthly aggregation
        monthly_sales = self.df.groupby(['Year', 'Month']).agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
        monthly_sales = monthly_sales.sort_values('Date')
        
        # Simple trend analysis
        monthly_sales['MonthNumber'] = range(len(monthly_sales))
        
        # Calculate trend (simple linear regression)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            monthly_sales['MonthNumber'], monthly_sales['TotalAmount']
        )
        
        # Seasonal pattern (average by month)
        seasonal_pattern = self.df.groupby('Month')['TotalAmount'].sum()
        seasonal_pattern = seasonal_pattern / seasonal_pattern.mean()  # Normalize
        
        # Simple forecast for next 6 months
        last_month_num = monthly_sales['MonthNumber'].max()
        forecast_months = []
        
        for i in range(1, 7):  # Next 6 months
            month_num = last_month_num + i
            trend_value = slope * month_num + intercept
            
            # Get seasonal factor
            future_month = ((monthly_sales['Month'].iloc[-1] + i - 1) % 12) + 1
            seasonal_factor = seasonal_pattern.loc[future_month]
            
            forecast_value = trend_value * seasonal_factor
            forecast_months.append({
                'MonthNumber': month_num,
                'ForecastRevenue': forecast_value,
                'Month': future_month
            })
        
        forecast_df = pd.DataFrame(forecast_months)
        
        # Visualization
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=monthly_sales['Date'],
            y=monthly_sales['TotalAmount'],
            mode='lines+markers',
            name='Historical Revenue',
            line=dict(color='blue', width=2)
        ))
        
        # Trend line
        trend_line = slope * monthly_sales['MonthNumber'] + intercept
        fig.add_trace(go.Scatter(
            x=monthly_sales['Date'],
            y=trend_line,
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        ))
        
        # Forecast (approximate dates)
        last_date = monthly_sales['Date'].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=6,
            freq='M'
        )
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_df['ForecastRevenue'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='green', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title='Sales Trend Analysis & 6-Month Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue (£)',
            height=500
        )
        fig.show()
        
        print(f"\n Forecasting Insights:")
        print(f"  Monthly Growth Rate: {slope:.2f} £/month")
        print(f"  Trend R²: {r_value**2:.3f}")
        print(f"  6-Month Forecast Total: £{forecast_df['ForecastRevenue'].sum():,.2f}")
        
        return forecast_df
    
    def generate_business_intelligence_report(self):
        """Generate comprehensive business intelligence report"""
        print("\n" + "="*60)
        print(" BUSINESS INTELLIGENCE REPORT")
        print("="*60)
        
        # Execute all advanced analyses
        cohort_analysis = self.cohort_analysis()
        market_basket = self.market_basket_analysis()
        clv_analysis = self.customer_lifetime_value()
        sales_forecast = self.seasonal_forecasting()
        
        # Strategic recommendations
        print(f"\n STRATEGIC RECOMMENDATIONS:")
        print(f"  1. Focus on customer retention - average Period 1 retention is key")
        print(f"  2. Leverage product associations for cross-selling opportunities")
        print(f"  3. Prioritize high CLV customers for personalized marketing")
        print(f"  4. Plan inventory based on seasonal forecasting patterns")
        
        return {
            'cohort_analysis': cohort_analysis,
            'market_basket': market_basket,
            'clv_analysis': clv_analysis,
            'sales_forecast': sales_forecast
        }

# Usage Example
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent 
    input_path = base_dir / "data" / "processed" / "online_retail_cleaned.csv"

    bi = BusinessInsights(input_path)
    comprehensive_bi_report = bi.generate_business_intelligence_report()
