import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EcommerceDataCleaner:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.cleaning_report = {}
        
    def load_data(self):
        """Load raw data and create initial report"""
        self.df = pd.read_csv(self.data_path, encoding='ISO-8859-1')
        
        # Initial data overview
        self.cleaning_report['initial_shape'] = self.df.shape
        self.cleaning_report['initial_memory'] = f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        
        print(f"Data loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        print(f"Memory usage: {self.cleaning_report['initial_memory']}")
        
        return self.df
    
    def analyze_data_quality(self):
        """Comprehensive data quality analysis"""
        print("\n" + "="*50)
        print("DATA QUALITY ANALYSIS")
        print("="*50)
        
        # Basic info
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Date Range: {self.df['InvoiceDate'].min()} to {self.df['InvoiceDate'].max()}")
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        print("\nMissing Values Analysis:")
        for col in self.df.columns:
            if missing_data[col] > 0:
                print(f"  {col}: {missing_data[col]:,} ({missing_percentage[col]:.2f}%)")
        
        # Data types
        print("\nData Types:")
        print(self.df.dtypes)
        
        # Unique values
        print("\nUnique Values:")
        for col in self.df.columns:
            print(f"  {col}: {self.df[col].nunique():,}")
        
        # Statistical summary for numerical columns
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        # Store in report
        self.cleaning_report['missing_data'] = missing_data.to_dict()
        self.cleaning_report['data_types'] = self.df.dtypes.to_dict()
        
    def clean_invoice_data(self):
        """Clean invoice-related fields"""
        print("\nCleaning Invoice Data...")
        
        initial_rows = len(self.df)
        
        # Remove cancelled orders (InvoiceNo starting with 'C')
        cancelled_orders = self.df[self.df['InvoiceNo'].astype(str).str.startswith('C')]
        print(f"  Found {len(cancelled_orders):,} cancelled orders ({len(cancelled_orders)/initial_rows*100:.2f}%)")
        
        self.df = self.df[~self.df['InvoiceNo'].astype(str).str.startswith('C')]
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        
        # Extract time features
        self.df['Year'] = self.df['InvoiceDate'].dt.year
        self.df['Month'] = self.df['InvoiceDate'].dt.month
        self.df['Day'] = self.df['InvoiceDate'].dt.day
        self.df['Hour'] = self.df['InvoiceDate'].dt.hour
        self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
        self.df['DayName'] = self.df['InvoiceDate'].dt.day_name()
        self.df['MonthName'] = self.df['InvoiceDate'].dt.month_name()
        
        print(f"  Removed {initial_rows - len(self.df):,} cancelled orders")
        print(f"  Added time-based features")
        
    def clean_product_data(self):
        """Clean product-related fields"""
        print("\nCleaning Product Data...")
        
        initial_missing_desc = self.df['Description'].isnull().sum()
        self.df = self.df.dropna(subset=['Description'])
        
        self.df['Description'] = self.df['Description'].str.strip()
        self.df['Description'] = self.df['Description'].str.upper()
        self.df['Description'] = self.df['Description'].str.replace(r'[^\w\s]', ' ', regex=True)
        self.df['Description'] = self.df['Description'].str.replace(r'\s+', ' ', regex=True)
        
        def categorize_product(description):
            description = description.lower()
            if any(word in description for word in ['bag', 'tote', 'shopping']):
                return 'Bags & Accessories'
            elif any(word in description for word in ['candle', 'holder', 'light']):
                return 'Home Decor'
            elif any(word in description for word in ['christmas', 'xmas', 'decoration']):
                return 'Seasonal'
            elif any(word in description for word in ['vintage', 'antique', 'craft']):
                return 'Vintage & Crafts'
            elif any(word in description for word in ['kitchen', 'cup', 'mug', 'plate']):
                return 'Kitchen & Dining'
            elif any(word in description for word in ['garden', 'outdoor', 'plant']):
                return 'Garden & Outdoor'
            else:
                return 'Other'
        
        self.df['ProductCategory'] = self.df['Description'].apply(categorize_product)
        
        print(f"  Removed {initial_missing_desc:,} rows with missing descriptions")
        print(f"  Standardized product descriptions")
        print(f"  Created product categories")
        
    def clean_financial_data(self):
        """Clean quantity and price data"""
        print("\nCleaning Financial Data...")
        
        initial_rows = len(self.df)
        zero_quantity = len(self.df[self.df['Quantity'] <= 0])
        print(f"  Found {zero_quantity:,} transactions with quantity ≤ 0")
        
        self.df = self.df[self.df['Quantity'] > 0]
        zero_price = len(self.df[self.df['UnitPrice'] <= 0])
        self.df = self.df[self.df['UnitPrice'] > 0]
        
        self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']
        
        def remove_outliers_iqr(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        before_outlier_removal = len(self.df)
        self.df = remove_outliers_iqr(self.df, 'Quantity')
        self.df = remove_outliers_iqr(self.df, 'UnitPrice')
        after_outlier_removal = len(self.df)
        
        print(f"  Removed {zero_quantity:,} zero/negative quantity transactions")
        print(f"  Removed {zero_price:,} zero/negative price transactions")
        print(f"  Removed {before_outlier_removal - after_outlier_removal:,} outlier transactions")
        print(f"  Created TotalAmount column")
        
    def clean_customer_data(self):
        """Clean customer-related data"""
        print("\nCleaning Customer Data...")
        
        missing_customers = self.df['CustomerID'].isnull().sum()
        print(f"  Found {missing_customers:,} transactions without CustomerID")
        
        df_with_customers = self.df.dropna(subset=['CustomerID'])
        self.df['HasCustomerID'] = ~self.df['CustomerID'].isnull()
        
        self.df.loc[~self.df['CustomerID'].isnull(), 'CustomerID'] = \
            self.df.loc[~self.df['CustomerID'].isnull(), 'CustomerID'].astype(int)
        
        self.df['Country'] = self.df['Country'].str.strip().str.title()
        
        print(f"  Flagged {missing_customers:,} transactions without CustomerID")
        print(f"  Standardized country names")
        print(f"  Available customers for analysis: {self.df['CustomerID'].nunique():,}")
        
    def create_derived_features(self):
        """Create business-relevant derived features"""
        print("\nCreating Derived Features...")
        
        self.df['TransactionValue'] = self.df.groupby('InvoiceNo')['TotalAmount'].transform('sum')
        self.df['ItemsPerTransaction'] = self.df.groupby('InvoiceNo')['Quantity'].transform('sum')
        self.df['UniqueProductsPerTransaction'] = self.df.groupby('InvoiceNo')['StockCode'].transform('nunique')
        self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6])
        
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        
        self.df['Season'] = self.df['Month'].apply(get_season)
        
        def get_business_period(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        
        self.df['BusinessPeriod'] = self.df['Hour'].apply(get_business_period)
        
        print(f"  Created transaction-level aggregations")
        print(f"  Added seasonal and time-based features")
        
    def generate_cleaning_summary(self):
        """Generate comprehensive cleaning report"""
        print("\n" + "="*50)
        print("DATA CLEANING SUMMARY")
        print("="*50)
        
        final_shape = self.df.shape
        final_memory = f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        
        print(f"Initial Shape: {self.cleaning_report['initial_shape']}")
        print(f"Final Shape: {final_shape}")
        print(f"Rows Removed: {self.cleaning_report['initial_shape'][0] - final_shape[0]:,}")
        print(f"Data Retention: {(final_shape[0]/self.cleaning_report['initial_shape'][0]*100):.2f}%")
        print(f"Memory Usage: {self.cleaning_report['initial_memory']} → {final_memory}")
        
        print("\nData Quality Metrics:")
        print(f"  Missing Values: {self.df.isnull().sum().sum():,}")
        print(f"  Duplicate Rows: {self.df.duplicated().sum():,}")
        print(f"  Unique Customers: {self.df['CustomerID'].nunique():,}")
        print(f"  Unique Products: {self.df['StockCode'].nunique():,}")
        print(f"  Date Range: {self.df['InvoiceDate'].min()} to {self.df['InvoiceDate'].max()}")
        print(f"  Countries: {self.df['Country'].nunique()}")
        
        self.cleaning_report.update({
            'final_shape': final_shape,
            'final_memory': final_memory,
            'data_quality': {
                'missing_values': self.df.isnull().sum().sum(),
                'duplicates': self.df.duplicated().sum(),
                'unique_customers': self.df['CustomerID'].nunique(),
                'unique_products': self.df['StockCode'].nunique(),
                'countries': self.df['Country'].nunique()
            }
        })
        
    def save_cleaned_data(self, output_path):
        """Save cleaned dataset"""
        self.df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
        
        import json
        report_path = Path(str(output_path).replace('.csv', '_cleaning_report.json'))
        with open(report_path, 'w') as f:
            json.dump(self.cleaning_report, f, indent=2, default=str)
        print(f"Cleaning report saved to: {report_path}")
        
        return self.df
    
    def full_cleaning_pipeline(self, output_path):
        """Execute complete cleaning pipeline"""
        print("Starting Complete Data Cleaning Pipeline...")
        
        self.load_data()
        self.analyze_data_quality()
        self.clean_invoice_data()
        self.clean_product_data()
        self.clean_financial_data()
        self.clean_customer_data()
        self.create_derived_features()
        self.generate_cleaning_summary()
        
        cleaned_df = self.save_cleaned_data(output_path)
        
        print("\nData cleaning pipeline completed successfully.")
        return cleaned_df

# Usage Example
if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "online_retail.csv"
    output_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "online_retail_cleaned.csv"

    cleaner = EcommerceDataCleaner(data_path)
    cleaned_data = cleaner.full_cleaning_pipeline(output_path)
