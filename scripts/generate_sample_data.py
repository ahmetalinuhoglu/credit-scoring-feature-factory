"""
Sample Data Generator

Generates realistic synthetic data for testing the credit scoring pipeline.
Creates sample applications and credit bureau data with realistic distributions.
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd


# Seed for reproducibility
RANDOM_SEED = 42

# Product types with their characteristics
PRODUCT_TYPES = {
    'INSTALLMENT_LOAN': {
        'weight': 0.40,  # 40% of records
        'default_rate': 0.06,
        'amount_range': (5000, 100000),
        'amount_mean': 25000,
        'amount_std': 20000,
        'duration_range': (12, 120),  # 1-10 years in months
        'duration_mean': 48,
        'interest_rate': 0.18  # Annual interest rate
    },
    'CASH_FACILITY': {
        'weight': 0.30,  # 30% of records
        'default_rate': 0.08,
        'amount_range': (1000, 50000),
        'amount_mean': 15000,
        'amount_std': 12000,
        'duration_range': None,  # Revolving - no fixed duration
        'duration_mean': None,
        'interest_rate': 0.24  # Annual interest rate (revolving)
    },
    'INSTALLMENT_SALE': {
        'weight': 0.15,  # 15% of records
        'default_rate': 0.20,
        'amount_range': (500, 30000),
        'amount_mean': 8000,
        'amount_std': 7000,
        'duration_range': (3, 48),  # 3 to 48 months
        'duration_mean': 18,
        'interest_rate': 0.22  # Annual interest rate
    },
    'MORTGAGE': {
        'weight': 0.10,  # 10% of records
        'default_rate': 0.01,
        'amount_range': (50000, 500000),
        'amount_mean': 200000,
        'amount_std': 100000,
        'duration_range': (60, 600),  # 5-50 years in months
        'duration_mean': 240,  # 20 years
        'interest_rate': 0.12  # Annual interest rate (lower for mortgage)
    },
    'NON_AUTH_OVERDRAFT': {
        'weight': 0.03,  # 3% of records
        'default_rate': 0.0,  # Not a credit product
        'amount_range': (100, 5000),
        'amount_mean': 1000,
        'amount_std': 800,
        'duration_range': None,
        'duration_mean': None,
        'interest_rate': None
    },
    'OVERLIMIT': {
        'weight': 0.02,  # 2% of records
        'default_rate': 0.0,  # Not a credit product
        'amount_range': (100, 3000),
        'amount_mean': 500,
        'amount_std': 400,
        'duration_range': None,
        'duration_mean': None,
        'interest_rate': None
    }
}

# Credit products (excluding non-credit)
CREDIT_PRODUCTS = ['INSTALLMENT_LOAN', 'CASH_FACILITY', 'INSTALLMENT_SALE', 'MORTGAGE']


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def generate_id(prefix: str, index: int) -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}_{index:08d}"


def random_date(start: datetime, end: datetime) -> datetime:
    """Generate a random date between start and end."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def generate_amount(product_type: str) -> float:
    """Generate a credit amount based on product type distribution."""
    config = PRODUCT_TYPES[product_type]
    
    # Generate from truncated normal distribution
    amount = np.random.normal(config['amount_mean'], config['amount_std'])
    
    # Clip to range
    min_amount, max_amount = config['amount_range']
    amount = np.clip(amount, min_amount, max_amount)
    
    return round(amount, 2)


def generate_duration(product_type: str) -> Optional[int]:
    """Generate credit duration in months based on product type."""
    config = PRODUCT_TYPES[product_type]
    
    if config['duration_range'] is None:
        return None  # Revolving products have no fixed duration
    
    min_months, max_months = config['duration_range']
    mean_months = config['duration_mean']
    std_months = (max_months - min_months) / 4  # Reasonable spread
    
    duration = int(np.random.normal(mean_months, std_months))
    duration = np.clip(duration, min_months, max_months)
    
    return int(duration)


def calculate_monthly_payment(amount: float, duration_months: Optional[int], product_type: str) -> Optional[float]:
    """
    Calculate monthly payment using standard amortization formula.
    
    PMT = P * [r(1+r)^n] / [(1+r)^n - 1]
    Where:
        P = Principal (amount)
        r = Monthly interest rate
        n = Number of months
    """
    config = PRODUCT_TYPES[product_type]
    
    if duration_months is None or config['interest_rate'] is None:
        return None  # Revolving products
    
    annual_rate = config['interest_rate']
    monthly_rate = annual_rate / 12
    n = duration_months
    
    if monthly_rate == 0:
        return round(amount / n, 2)
    
    payment = amount * (monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)
    return round(payment, 2)


def assign_product_type() -> str:
    """Assign a product type based on weights."""
    products = list(PRODUCT_TYPES.keys())
    weights = [PRODUCT_TYPES[p]['weight'] for p in products]
    return np.random.choice(products, p=weights)


def should_default(product_type: str, customer_risk_score: float) -> bool:
    """
    Determine if a credit should default based on product type and customer risk.
    
    Args:
        product_type: Type of product
        customer_risk_score: Customer's risk score (0-1, higher = riskier)
        
    Returns:
        True if should default
    """
    base_rate = PRODUCT_TYPES[product_type]['default_rate']
    
    if base_rate == 0:  # Non-credit products
        return False
    
    # Adjust rate based on customer risk
    # Risk score of 0.5 = average, higher = more likely to default
    adjusted_rate = base_rate * (0.5 + customer_risk_score)
    adjusted_rate = min(adjusted_rate, 0.5)  # Cap at 50%
    
    return random.random() < adjusted_rate


def generate_applications(
    n_applications: int,
    start_date: datetime,
    end_date: datetime,
    joint_application_rate: float = 0.15
) -> pd.DataFrame:
    """
    Generate sample application data.
    
    Args:
        n_applications: Number of applications to generate
        start_date: Start date for applications
        end_date: End date for applications
        joint_application_rate: Rate of joint applications
        
    Returns:
        DataFrame with application data
    """
    records = []
    customer_id = 0
    
    for app_idx in range(n_applications):
        application_id = generate_id("APP", app_idx)
        application_date = random_date(start_date, end_date)
        
        # Determine if joint application
        is_joint = random.random() < joint_application_rate
        n_applicants = 2 if is_joint else 1
        
        # Generate customer risk score for this application
        # This will influence both primary and co-applicant
        base_risk = np.random.beta(2, 5)  # Skewed towards lower risk
        
        for applicant_idx in range(n_applicants):
            customer_id += 1
            applicant_type = "PRIMARY" if applicant_idx == 0 else "CO_APPLICANT"
            
            # Co-applicants tend to have slightly lower risk
            if applicant_type == "CO_APPLICANT":
                risk_score = base_risk * 0.9
            else:
                risk_score = base_risk
            
            records.append({
                'application_id': application_id,
                'customer_id': generate_id("CUST", customer_id),
                'applicant_type': applicant_type,
                'application_date': application_date.strftime('%Y-%m-%d'),
                'target': None,  # Will be set later based on credit history
                '_risk_score': risk_score  # Internal, for generating credit bureau data
            })
    
    return pd.DataFrame(records)


def generate_credit_bureau(
    applications_df: pd.DataFrame,
    avg_credits_per_customer: float = 5.0,
    credit_history_years: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate sample credit bureau data for each customer.
    
    Args:
        applications_df: DataFrame with application data
        avg_credits_per_customer: Average number of credits per customer
        credit_history_years: Maximum age of credit history in years
        
    Returns:
        Tuple of (credit_bureau_df, updated_applications_df)
    """
    credit_records = []
    applications_df = applications_df.copy()
    
    for _, row in applications_df.iterrows():
        application_id = row['application_id']
        customer_id = row['customer_id']
        application_date = datetime.strptime(row['application_date'], '%Y-%m-%d')
        risk_score = row['_risk_score']
        
        # Number of credits for this customer (from Poisson distribution)
        n_credits = max(1, np.random.poisson(avg_credits_per_customer))
        
        # Track defaults for this customer
        has_current_default = False
        ever_defaulted = False
        
        for _ in range(n_credits):
            product_type = assign_product_type()
            amount = generate_amount(product_type)
            
            # Opening date (before application date)
            max_history = timedelta(days=credit_history_years * 365)
            earliest_opening = application_date - max_history
            opening_date = random_date(earliest_opening, application_date - timedelta(days=30))
            
            # Determine default status
            default_date = None
            recovery_date = None
            
            if should_default(product_type, risk_score):
                ever_defaulted = True
                
                # Default happens between opening and application date
                days_to_default = random.randint(30, (application_date - opening_date).days)
                default_date = opening_date + timedelta(days=days_to_default)
                
                # Recovery chance (50% of defaults recover)
                if random.random() < 0.5:
                    days_to_recovery = random.randint(30, 365)
                    recovery_date = default_date + timedelta(days=days_to_recovery)
                    
                    # If recovery is after application date, it's still defaulted at time of application
                    if recovery_date > application_date:
                        recovery_date = None
                        has_current_default = True
                else:
                    has_current_default = True
            
            # Generate duration and monthly payment
            duration_months = generate_duration(product_type)
            monthly_payment = calculate_monthly_payment(amount, duration_months, product_type)
            
            # Calculate closure_date for term products (if paid off normally)
            closure_date = None
            if duration_months is not None and default_date is None:
                # Credit closes when term ends
                planned_closure = opening_date + timedelta(days=int(duration_months * 30.44))
                
                # Only set closure_date if credit has actually closed before application
                if planned_closure < application_date:
                    closure_date = planned_closure
                    
                    # === 3-MONTH DELETION RULE ===
                    # Credits closed more than 90 days ago are deleted from bureau
                    days_since_closure = (application_date - closure_date).days
                    if days_since_closure > 90:
                        # Skip this credit - it would have been deleted
                        continue
            
            credit_records.append({
                'application_id': application_id,
                'customer_id': customer_id,
                'product_type': product_type,
                'total_amount': amount,
                'duration_months': duration_months,
                'monthly_payment': monthly_payment,
                'opening_date': opening_date.strftime('%Y-%m-%d'),
                'closure_date': closure_date.strftime('%Y-%m-%d') if closure_date else None,
                'default_date': default_date.strftime('%Y-%m-%d') if default_date else None,
                'recovery_date': recovery_date.strftime('%Y-%m-%d') if recovery_date else None
            })
        
        # Set target based on defaults
        # Higher risk customers with defaults are more likely to default in future
        if has_current_default:
            target = 1 if random.random() < 0.7 else 0  # 70% chance if currently defaulted
        elif ever_defaulted:
            target = 1 if random.random() < 0.3 else 0  # 30% chance if ever defaulted
        else:
            target = 1 if random.random() < risk_score * 0.15 else 0  # Base risk
        
        # Update target in applications
        applications_df.loc[
            (applications_df['application_id'] == application_id) & 
            (applications_df['customer_id'] == customer_id),
            'target'
        ] = target
    
    # Remove internal risk score column
    applications_df = applications_df.drop(columns=['_risk_score'])
    
    return pd.DataFrame(credit_records), applications_df


def generate_sample_data(
    n_applications: int = 10000,
    output_dir: str = "data/sample",
    seed: int = RANDOM_SEED
) -> Dict[str, str]:
    """
    Generate complete sample dataset.
    
    Args:
        n_applications: Number of applications to generate
        output_dir: Output directory for CSV files
        seed: Random seed
        
    Returns:
        Dictionary with paths to generated files
    """
    set_seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {n_applications} applications...")
    
    # Date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 1, 31)
    
    # Generate applications
    applications_df = generate_applications(
        n_applications=n_applications,
        start_date=start_date,
        end_date=end_date,
        joint_application_rate=0.15
    )
    
    print(f"Generated {len(applications_df)} customer records (including joint applications)")
    
    # Generate credit bureau data
    print("Generating credit bureau data...")
    credit_bureau_df, applications_df = generate_credit_bureau(
        applications_df,
        avg_credits_per_customer=5.0,
        credit_history_years=10
    )
    
    print(f"Generated {len(credit_bureau_df)} credit bureau records")
    
    # Save to CSV
    applications_path = output_path / "sample_applications.csv"
    credit_bureau_path = output_path / "sample_credit_bureau.csv"
    
    applications_df.to_csv(applications_path, index=False)
    credit_bureau_df.to_csv(credit_bureau_path, index=False)
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("SAMPLE DATA SUMMARY")
    print("="*60)
    
    print(f"\nApplications:")
    print(f"  Total records: {len(applications_df):,}")
    print(f"  Unique applications: {applications_df['application_id'].nunique():,}")
    print(f"  Joint applications: {(applications_df['applicant_type'] == 'CO_APPLICANT').sum():,}")
    print(f"  Target distribution:")
    print(f"    - Non-default (0): {(applications_df['target'] == 0).sum():,} ({(applications_df['target'] == 0).mean()*100:.1f}%)")
    print(f"    - Default (1): {(applications_df['target'] == 1).sum():,} ({(applications_df['target'] == 1).mean()*100:.1f}%)")
    
    print(f"\nCredit Bureau:")
    print(f"  Total records: {len(credit_bureau_df):,}")
    print(f"  Product type distribution:")
    for product in PRODUCT_TYPES.keys():
        count = (credit_bureau_df['product_type'] == product).sum()
        pct = count / len(credit_bureau_df) * 100
        print(f"    - {product}: {count:,} ({pct:.1f}%)")
    
    # Default statistics for credit products
    credit_only = credit_bureau_df[credit_bureau_df['product_type'].isin(CREDIT_PRODUCTS)]
    print(f"\n  Default statistics (credit products only):")
    print(f"    - Total credit records: {len(credit_only):,}")
    print(f"    - Records with default: {credit_only['default_date'].notna().sum():,}")
    print(f"    - Records recovered: {credit_only['recovery_date'].notna().sum():,}")
    
    for product in CREDIT_PRODUCTS:
        product_df = credit_bureau_df[credit_bureau_df['product_type'] == product]
        actual_rate = product_df['default_date'].notna().mean() * 100
        expected_rate = PRODUCT_TYPES[product]['default_rate'] * 100
        print(f"    - {product}: {actual_rate:.1f}% (expected: {expected_rate:.1f}%)")
    
    print(f"\nFiles saved:")
    print(f"  - {applications_path}")
    print(f"  - {credit_bureau_path}")
    
    # Create README
    readme_content = generate_readme(applications_df, credit_bureau_df)
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  - {readme_path}")
    
    return {
        'applications': str(applications_path),
        'credit_bureau': str(credit_bureau_path),
        'readme': str(readme_path)
    }


def generate_readme(applications_df: pd.DataFrame, credit_bureau_df: pd.DataFrame) -> str:
    """Generate README for sample data."""
    
    target_dist = applications_df['target'].value_counts()
    product_dist = credit_bureau_df['product_type'].value_counts()
    
    return f"""# Sample Credit Bureau Data

This directory contains synthetic sample data for testing the credit scoring pipeline.

## Data Description

### Applications (`sample_applications.csv`)

Contains credit application data with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `application_id` | string | Unique application identifier |
| `customer_id` | string | Unique customer identifier |
| `applicant_type` | string | PRIMARY or CO_APPLICANT |
| `application_date` | date | Date of application |
| `target` | int | 12-month default flag (0/1) |

**Statistics:**
- Total records: {len(applications_df):,}
- Unique applications: {applications_df['application_id'].nunique():,}
- Default rate: {applications_df['target'].mean()*100:.1f}%

### Credit Bureau (`sample_credit_bureau.csv`)

Contains credit history at the time of application:

| Column | Type | Description |
|--------|------|-------------|
| `application_id` | string | Application ID (FK) |
| `customer_id` | string | Customer ID |
| `product_type` | string | Credit product type |
| `total_amount` | float | Credit amount |
| `opening_date` | date | Credit opening date |
| `default_date` | date | Default date (nullable) |
| `recovery_date` | date | Recovery date (nullable) |

**Product Types:**
{chr(10).join([f'- `{p}`: {c:,} records ({c/len(credit_bureau_df)*100:.1f}%)' for p, c in product_dist.items()])}

## Usage

```python
import pandas as pd

# Load data
applications = pd.read_csv('sample_applications.csv')
credit_bureau = pd.read_csv('sample_credit_bureau.csv')

# Join on customer
merged = applications.merge(
    credit_bureau, 
    on=['application_id', 'customer_id'],
    how='left'
)
```

## Notes

- This is **synthetic data** generated for testing purposes
- Default rates are approximate and may vary slightly due to random generation
- Joint applications share the same `application_id` but have different `customer_id`
- `NON_AUTH_OVERDRAFT` and `OVERLIMIT` are not credit products (no default rates)

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Generate sample data for credit scoring pipeline"
    )
    parser.add_argument(
        '-n', '--n-applications',
        type=int,
        default=10000,
        help='Number of applications to generate (default: 10000)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='data/sample',
        help='Output directory (default: data/sample)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed (default: {RANDOM_SEED})'
    )
    
    args = parser.parse_args()
    
    generate_sample_data(
        n_applications=args.n_applications,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
