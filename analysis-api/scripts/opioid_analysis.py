import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

def analyze_medication_transitions(df: pd.DataFrame) -> Dict:
    """
    Analyzes patient medication transitions and adherence patterns.
    
    Parameters:
    df: DataFrame with columns:
        - patient_id
        - medication_name
        - prescription_date
        - dosage
        - next_appointment
        - attendance_status
    
    Returns:
    Dictionary containing transition matrices and adherence metrics
    """
    # Sort data by patient and date
    df = df.sort_values(['patient_id', 'prescription_date'])
    
    # Calculate medication transitions
    df['prev_medication'] = df.groupby('patient_id')['medication_name'].shift(1)
    
    # Create transition matrix
    transition_matrix = pd.crosstab(
        df['prev_medication'], 
        df['medication_name'], 
        normalize='index'
    )
    
    # Calculate adherence metrics
    adherence_metrics = {
        'appointment_adherence': (
            df['attendance_status'].value_counts(normalize=True)
        ),
        'avg_days_between_visits': (
            df.groupby('patient_id')['prescription_date']
            .diff()
            .mean()
            .days
        )
    }
    
    # Identify common transition patterns
    transitions = df[df['prev_medication'].notna()].groupby(
        ['prev_medication', 'medication_name']
    ).size().reset_index(name='count')
    
    transitions = transitions.sort_values('count', ascending=False)
    
    return {
        'transition_matrix': transition_matrix,
        'adherence_metrics': adherence_metrics,
        'common_transitions': transitions.head(10)
    }

def calculate_risk_metrics(df: pd.DataFrame, 
                         high_risk_threshold: int = 90) -> pd.DataFrame:
    """
    Calculates patient risk metrics based on prescription patterns.
    
    Parameters:
    df: DataFrame with patient prescription data
    high_risk_threshold: Days threshold for high-risk designation
    
    Returns:
    DataFrame with risk metrics per patient
    """
    # Calculate days between prescriptions
    df = df.sort_values(['patient_id', 'prescription_date'])
    df['days_between_prescriptions'] = df.groupby('patient_id')['prescription_date'].diff().dt.days
    
    # Calculate risk metrics per patient
    risk_metrics = df.groupby('patient_id').agg({
        'days_between_prescriptions': ['mean', 'std', 'max'],
        'dosage': ['mean', 'max', lambda x: x.diff().abs().mean()],
        'attendance_status': lambda x: (x == 'Missed').mean()
    }).reset_index()
    
    # Flatten column names
    risk_metrics.columns = [
        'patient_id', 'avg_days_between_rx', 'std_days_between_rx', 
        'max_days_between_rx', 'avg_dosage', 'max_dosage', 
        'avg_dosage_change', 'missed_appointment_rate'
    ]
    
    # Calculate risk score (example method - should be calibrated to clinic needs)
    risk_metrics['risk_score'] = (
        (risk_metrics['max_days_between_rx'] > high_risk_threshold).astype(int) * 3 +
        (risk_metrics['missed_appointment_rate'] > 0.2).astype(int) * 2 +
        (risk_metrics['avg_dosage_change'] > risk_metrics['avg_dosage_change'].median()).astype(int)
    )
    
    return risk_metrics

def generate_summary_report(df: pd.DataFrame, 
                          start_date: datetime,
                          end_date: datetime) -> Dict:
    """
    Generates a comprehensive summary report for the specified date range.
    
    Parameters:
    df: DataFrame with patient prescription data
    start_date: Start date for the report period
    end_date: End date for the report period
    
    Returns:
    Dictionary containing summary metrics and analyses
    """
    # Filter data for date range
    mask = (df['prescription_date'] >= start_date) & (df['prescription_date'] <= end_date)
    period_data = df[mask].copy()
    
    # Calculate basic metrics
    total_patients = period_data['patient_id'].nunique()
    total_prescriptions = len(period_data)
    
    # Medication distribution
    med_distribution = period_data['medication_name'].value_counts()
    
    # Average dosage by medication
    avg_dosage = period_data.groupby('medication_name')['dosage'].agg([
        'mean', 'median', 'std'
    ])
    
    # Calculate patient retention
    retention_data = period_data.groupby('patient_id').agg({
        'prescription_date': ['min', 'max']
    })
    retention_data.columns = ['first_visit', 'last_visit']
    retention_data['days_retained'] = (
        retention_data['last_visit'] - retention_data['first_visit']
    ).dt.days
    
    return {
        'period_summary': {
            'start_date': start_date,
            'end_date': end_date,
            'total_patients': total_patients,
            'total_prescriptions': total_prescriptions,
            'avg_prescriptions_per_patient': total_prescriptions / total_patients
        },
        'medication_distribution': med_distribution,
        'dosage_statistics': avg_dosage,
        'retention_metrics': {
            'avg_retention_days': retention_data['days_retained'].mean(),
            'retention_distribution': retention_data['days_retained'].describe()
        }
    }

def plot_trends(df: pd.DataFrame, metric: str = 'prescriptions') -> plt.Figure:
    """
    Creates visualization of trends over time.
    
    Parameters:
    df: DataFrame with patient prescription data
    metric: Type of trend to plot ('prescriptions' or 'dosage')
    
    Returns:
    Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if metric == 'prescriptions':
        # Plot prescription counts over time
        daily_rx = df.groupby('prescription_date').size()
        daily_rx.plot(ax=ax, title='Daily Prescription Volume')
        ax.set_ylabel('Number of Prescriptions')
        
    elif metric == 'dosage':
        # Plot average dosage trends by medication
        dosage_trends = df.pivot_table(
            index='prescription_date',
            columns='medication_name',
            values='dosage',
            aggfunc='mean'
        )
        dosage_trends.plot(ax=ax, title='Average Dosage Trends by Medication')
        ax.set_ylabel('Average Dosage')
    
    ax.set_xlabel('Date')
    plt.tight_layout()
    
    return fig

# Example usage:
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'patient_id': range(100),
        'medication_name': np.random.choice(
            ['Methadone', 'Buprenorphine', 'Naltrexone'], 
            100
        ),
        'prescription_date': pd.date_range(
            start='2024-01-01', 
            periods=100, 
            freq='D'
        ),
        'dosage': np.random.normal(50, 10, 100),
        'next_appointment': pd.date_range(
            start='2024-02-01', 
            periods=100, 
            freq='D'
        ),
        'attendance_status': np.random.choice(
            ['Attended', 'Missed'], 
            100, 
            p=[0.8, 0.2]
        )
    })
    
    # Run analyses
    transitions = analyze_medication_transitions(sample_data)
    risks = calculate_risk_metrics(sample_data)
    summary = generate_summary_report(
        sample_data, 
        datetime(2024, 1, 1), 
        datetime(2024, 3, 1)
    )
    trend_plot = plot_trends(sample_data, 'prescriptions')
