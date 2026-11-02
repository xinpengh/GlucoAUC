#!/usr/bin/env python3
"""
Continuous Glucose Monitor (CGM) Analysis Script
Analyzes FreeStyle Libre data for glycemic control metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import signal, integrate
from scipy.stats import variation

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CGMAnalyzer:
    def __init__(self, csv_path):
        """Initialize with CSV file path"""
        # FreeStyle Libre CSVs have a header row, skip it
        self.df = pd.read_csv(csv_path, skiprows=1, low_memory=False)
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Clean and prepare data"""
        # Find datetime column (Device Timestamp for FreeStyle Libre)
        datetime_cols = [col for col in self.df.columns if 'timestamp' in col.lower() or 
                        ('device' in col.lower() and 'time' in col.lower())]
        
        if datetime_cols:
            self.df['datetime'] = pd.to_datetime(self.df[datetime_cols[0]], 
                                                  format='%d-%m-%Y %H:%M', 
                                                  errors='coerce')
        else:
            raise ValueError("Could not find datetime column. Column names: " + str(self.df.columns.tolist()))
        
        # Handle FreeStyle Libre's split glucose columns
        # Historic Glucose mmol/L (record type 0) and Scan Glucose mmol/L (record type 1)
        historic_col = [col for col in self.df.columns if 'historic' in col.lower() and 'glucose' in col.lower() and 'mmol' in col.lower()]
        scan_col = [col for col in self.df.columns if 'scan' in col.lower() and 'glucose' in col.lower() and 'mmol' in col.lower()]
        
        if historic_col and scan_col:
            # Merge both columns - use scan if available, otherwise historic
            self.df['glucose'] = pd.to_numeric(self.df[scan_col[0]], errors='coerce').fillna(
                                 pd.to_numeric(self.df[historic_col[0]], errors='coerce'))
            print(f"Found both Historic and Scan glucose columns - merging...")
        elif historic_col:
            self.df['glucose'] = pd.to_numeric(self.df[historic_col[0]], errors='coerce')
        elif scan_col:
            self.df['glucose'] = pd.to_numeric(self.df[scan_col[0]], errors='coerce')
        else:
            # Fallback to any glucose column
            glucose_cols = [col for col in self.df.columns if 'glucose' in col.lower() and 'mmol' in col.lower()]
            if glucose_cols:
                self.df['glucose'] = pd.to_numeric(self.df[glucose_cols[0]], errors='coerce')
            else:
                raise ValueError("Could not find glucose column. Column names: " + str(self.df.columns.tolist()))
        
        # Find record type column
        record_type_cols = [col for col in self.df.columns if 'record' in col.lower() and 'type' in col.lower()]
        if record_type_cols:
            self.df['record_type'] = pd.to_numeric(self.df[record_type_cols[0]], errors='coerce')
        else:
            self.df['record_type'] = None
        
        # Drop rows with missing datetime or glucose
        before_drop = len(self.df)
        self.df = self.df.dropna(subset=['datetime', 'glucose'])
        after_drop = len(self.df)
        
        if before_drop > after_drop:
            print(f"Removed {before_drop - after_drop} rows with missing datetime or glucose")
        
        # Filter for record types 0 (historic/automatic) and 1 (scan)
        if 'record_type' in self.df.columns and self.df['record_type'].notna().any():
            print(f"\nRecord type distribution:")
            print(self.df['record_type'].value_counts().sort_index())
            
            # Keep only types 0 and 1
            self.df = self.df[self.df['record_type'].isin([0, 1])].copy()
            print(f"\nFiltered to record types 0 and 1: {len(self.df)} measurements")
            
            # Sort by datetime
            self.df = self.df.sort_values('datetime').reset_index(drop=True)
            
            # Handle overlapping measurements
            # Priority: Type 1 (scan) > Type 0 (historic) for same timestamp
            # Remove duplicates within 1-minute windows (keep scans preferentially)
            self.df = self.df.sort_values(['datetime', 'record_type'], ascending=[True, False])
            
            # Remove duplicates within 1 minute
            self.df['datetime_rounded'] = self.df['datetime'].dt.round('1min')
            
            # Keep first occurrence (which is scan due to sorting)
            before_dedup = len(self.df)
            self.df = self.df.drop_duplicates(subset=['datetime_rounded'], keep='first')
            self.df = self.df.drop(columns=['datetime_rounded'])
            after_dedup = len(self.df)
            
            if before_dedup > after_dedup:
                print(f"Removed {before_dedup - after_dedup} duplicate measurements (preferring scans)")
        else:
            self.df = self.df.sort_values('datetime').reset_index(drop=True)
        
        print(f"\nFinal dataset: {len(self.df)} glucose measurements")
        print(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        print(f"Duration: {(self.df['datetime'].max() - self.df['datetime'].min()).days} days")
        print(f"Glucose range: {self.df['glucose'].min():.1f} - {self.df['glucose'].max():.1f} mmol/L")
        
    def calculate_summary_stats(self):
        """Calculate overall glycemic metrics"""
        glucose = self.df['glucose']
        
        stats = {
            'Mean Glucose (mmol/L)': glucose.mean(),
            'Median Glucose (mmol/L)': glucose.median(),
            'SD (mmol/L)': glucose.std(),
            'CV (%)': variation(glucose) * 100,
            'Min (mmol/L)': glucose.min(),
            'Max (mmol/L)': glucose.max(),
            'IQR (mmol/L)': glucose.quantile(0.75) - glucose.quantile(0.25),
        }
        
        # Time in range metrics
        stats['Time in Range 3.9-10.0 (%)'] = ((glucose >= 3.9) & (glucose <= 10.0)).mean() * 100
        stats['Time in Tight Range 3.9-7.8 (%)'] = ((glucose >= 3.9) & (glucose <= 7.8)).mean() * 100
        stats['Time Above 10.0 (%)'] = (glucose > 10.0).mean() * 100
        stats['Time Above 7.8 (%)'] = (glucose > 7.8).mean() * 100
        stats['Time Below 3.9 (%)'] = (glucose < 3.9).mean() * 100
        
        return stats
    
    def calculate_mage(self):
        """Mean Amplitude of Glycemic Excursions"""
        glucose = self.df['glucose'].values
        sd = self.df['glucose'].std()
        
        # Find peaks and troughs
        peaks, _ = signal.find_peaks(glucose, distance=12)  # 1 hour apart (5min intervals)
        troughs, _ = signal.find_peaks(-glucose, distance=12)
        
        # Calculate excursions > 1 SD
        excursions = []
        
        for peak in peaks:
            # Find nearest trough before peak
            before_troughs = troughs[troughs < peak]
            if len(before_troughs) > 0:
                trough = before_troughs[-1]
                excursion = glucose[peak] - glucose[trough]
                if excursion > sd:
                    excursions.append(excursion)
        
        mage = np.mean(excursions) if excursions else 0
        return mage
    
    def detect_spikes(self, baseline_window=30, spike_threshold=1.5, 
                      min_spike_duration=15, min_peak_duration=10, return_to_baseline_threshold=0.5):
        """
        Detect glucose spikes (e.g., post-prandial) with improved algorithm
        
        Parameters:
        - baseline_window: minutes to look back for baseline (default 30)
        - spike_threshold: mmol/L above baseline to consider a spike (default 1.5)
        - min_spike_duration: minimum duration above baseline in minutes (default 15)
        - min_peak_duration: minimum time at elevated levels in minutes (default 10)
        - return_to_baseline_threshold: how close to baseline before ending spike (default 0.5 mmol/L)
        
        Returns DataFrame with spike characteristics
        """
        glucose = self.df['glucose'].values
        times = self.df['datetime'].values
        
        # Calculate median time interval
        time_diffs = np.diff(pd.to_datetime(times).astype(np.int64) / 1e9 / 60)  # minutes
        median_interval = np.median(time_diffs[time_diffs > 0])
        print(f"Median measurement interval: {median_interval:.1f} minutes")
        
        spikes = []
        i = int(baseline_window / median_interval)  # Convert to number of readings
        
        min_readings = int(min_spike_duration / median_interval)
        
        in_spike = False
        spike_start_idx = None
        
        while i < len(glucose):
            # Calculate rolling baseline (minimum in window)
            baseline_start = max(0, i - int(baseline_window / median_interval))
            baseline = glucose[baseline_start:i].min() if baseline_start < i else glucose[i]
            
            current_elevation = glucose[i] - baseline
            
            # Start of spike detection
            if not in_spike and current_elevation >= spike_threshold:
                spike_start_idx = i
                spike_baseline = baseline
                in_spike = True
                
            # End of spike detection
            elif in_spike and current_elevation < return_to_baseline_threshold:
                spike_end_idx = i
                
                # Validate spike duration
                if spike_end_idx - spike_start_idx >= min_readings:
                    # Find peak within spike
                    spike_segment = glucose[spike_start_idx:spike_end_idx+1]
                    peak_offset = np.argmax(spike_segment)
                    peak_idx = spike_start_idx + peak_offset
                    peak_value = glucose[peak_idx]
                    
                    # Only count if peak is sufficiently elevated
                    if peak_value - spike_baseline >= spike_threshold:
                        # Calculate detailed metrics
                        spike_glucose = glucose[spike_start_idx:spike_end_idx+1]
                        spike_times_dt = pd.to_datetime(times[spike_start_idx:spike_end_idx+1])
                        
                        # Time differences in hours for AUC
                        time_hours = (spike_times_dt - spike_times_dt[0]).total_seconds() / 3600
                        
                        # Incremental AUC (trapezoidal integration above baseline)
                        iauc = integrate.trapezoid(np.maximum(spike_glucose - spike_baseline, 0), time_hours)
                        
                        # Duration metrics
                        total_duration = (spike_times_dt[-1] - spike_times_dt[0]).total_seconds() / 60
                        time_to_peak = (spike_times_dt[peak_offset] - spike_times_dt[0]).total_seconds() / 60
                        time_from_peak = total_duration - time_to_peak
                        
                        # Descent phase analysis (from peak to end)
                        if peak_idx < spike_end_idx:
                            descent_glucose = glucose[peak_idx:spike_end_idx+1]
                            descent_times_dt = pd.to_datetime(times[peak_idx:spike_end_idx+1])
                            descent_minutes = (descent_times_dt - descent_times_dt[0]).total_seconds() / 60
                            
                            # Calculate descent rate (negative slope)
                            if len(descent_minutes) > 1:
                                descent_rate = -np.polyfit(descent_minutes, descent_glucose, 1)[0]
                                
                                # Also calculate time to return to baseline + 20% threshold
                                threshold_value = spike_baseline + (peak_value - spike_baseline) * 0.2
                                below_threshold = descent_glucose < threshold_value
                                if np.any(below_threshold):
                                    recovery_idx = np.argmax(below_threshold)
                                    recovery_time = descent_minutes[recovery_idx]
                                else:
                                    recovery_time = descent_minutes[-1]
                            else:
                                descent_rate = 0
                                recovery_time = 0
                        else:
                            descent_rate = 0
                            recovery_time = 0
                        
                        # Calculate glucose clearance efficiency
                        # (peak - baseline) / time to clear
                        clearance_efficiency = (peak_value - spike_baseline) / recovery_time if recovery_time > 0 else 0
                        
                        spikes.append({
                            'start_time': times[spike_start_idx],
                            'end_time': times[spike_end_idx],
                            'peak_time': times[peak_idx],
                            'baseline': spike_baseline,
                            'peak_glucose': peak_value,
                            'delta': peak_value - spike_baseline,
                            'iauc': iauc,
                            'duration_min': total_duration,
                            'time_to_peak_min': time_to_peak,
                            'time_from_peak_min': time_from_peak,
                            'recovery_time_min': recovery_time,
                            'descent_rate': descent_rate,
                            'clearance_efficiency': clearance_efficiency,
                            'start_idx': spike_start_idx,
                            'end_idx': spike_end_idx,
                            'peak_idx': peak_idx
                        })
                
                # Reset spike tracking
                in_spike = False
                spike_start_idx = None
            
            i += 1
        
        return pd.DataFrame(spikes)
    
    def analyze_fasting_glucose(self):
        """Extract fasting glucose (e.g., 6-8 AM)"""
        self.df['hour'] = self.df['datetime'].dt.hour
        fasting = self.df[(self.df['hour'] >= 6) & (self.df['hour'] < 8)]
        
        if len(fasting) == 0:
            return None
        
        return {
            'Mean Fasting Glucose': fasting['glucose'].mean(),
            'Fasting Glucose SD': fasting['glucose'].std(),
            'Days Measured': fasting['datetime'].dt.date.nunique()
        }
    
    def plot_glucose_trace(self, figsize=(15, 6)):
        """Plot full glucose trace with target ranges"""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.df['datetime'], self.df['glucose'], linewidth=0.8, alpha=0.8)
        
        # Target ranges
        ax.axhspan(3.9, 7.8, alpha=0.1, color='green', label='Tight Range (3.9-7.8)')
        ax.axhspan(7.8, 10.0, alpha=0.1, color='yellow', label='Standard Range (7.8-10.0)')
        ax.axhline(3.9, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Hypoglycemia')
        ax.axhline(10.0, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Hyperglycemia')
        
        ax.set_xlabel('Date/Time')
        ax.set_ylabel('Glucose (mmol/L)')
        ax.set_title('Continuous Glucose Monitor Trace')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_daily_patterns(self, figsize=(12, 6)):
        """Plot average glucose by time of day"""
        self.df['time_only'] = self.df['datetime'].dt.hour + self.df['datetime'].dt.minute / 60
        
        # Group by time and calculate statistics
        hourly = self.df.groupby(self.df['datetime'].dt.hour)['glucose'].agg(['mean', 'std', 'count'])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(hourly.index, hourly['mean'], marker='o', linewidth=2, markersize=6)
        ax.fill_between(hourly.index, 
                        hourly['mean'] - hourly['std'], 
                        hourly['mean'] + hourly['std'],
                        alpha=0.3)
        
        ax.axhspan(3.9, 7.8, alpha=0.1, color='green')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Glucose (mmol/L)')
        ax.set_title('Average Glucose by Time of Day (±1 SD)')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_spikes(self, spikes_df, max_spikes=10, figsize=(15, 18)):
        """Plot individual glucose spikes"""
        if len(spikes_df) == 0:
            print("No spikes detected")
            return None
        
        # Plot up to max_spikes
        n_plots = min(len(spikes_df), max_spikes)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, (idx, spike) in enumerate(spikes_df.head(n_plots).iterrows()):
            ax = axes[i]
            
            # Extract spike data
            start_idx = spike['start_idx']
            end_idx = spike['end_idx']
            peak_idx = spike['peak_idx']
            
            spike_data = self.df.iloc[start_idx:end_idx+1]
            
            # Plot
            ax.plot(spike_data['datetime'], spike_data['glucose'], 
                   linewidth=2, marker='o', markersize=3)
            ax.axhline(spike['baseline'], color='green', linestyle='--', 
                      label=f"Baseline: {spike['baseline']:.1f}")
            ax.plot(spike_data.iloc[peak_idx - start_idx]['datetime'], 
                   spike['peak_glucose'], 'r*', markersize=15, label=f"Peak: {spike['peak_glucose']:.1f}")
            
            # Annotations
            ax.set_title(f"Spike #{i+1}: iAUC={spike['iauc']:.1f}, Duration={spike['duration_min']:.0f}min, "
                        f"Δ={spike['delta']:.1f} mmol/L", fontsize=10)
            ax.set_ylabel('Glucose (mmol/L)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            if i == n_plots - 1:
                ax.set_xlabel('Time')
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def analyze_spike_trends(self, spikes_df):
        """Analyze how spike characteristics change over time"""
        if len(spikes_df) == 0:
            return None
        
        spikes_df['date'] = pd.to_datetime(spikes_df['start_time']).dt.date
        spikes_df['week'] = pd.to_datetime(spikes_df['start_time']).dt.isocalendar().week
        
        # Weekly aggregation
        weekly = spikes_df.groupby('week').agg({
            'iauc': ['mean', 'std', 'count'],
            'descent_rate': 'mean',
            'recovery_time_min': 'mean',
            'delta': 'mean'
        }).reset_index()
        
        weekly.columns = ['week', 'mean_iauc', 'std_iauc', 'n_spikes', 
                         'mean_descent_rate', 'mean_recovery_time', 'mean_delta']
        
        return weekly
    
    def plot_spike_trends(self, spikes_df, figsize=(15, 10)):
        """Plot trends in insulin sensitivity metrics over time"""
        if len(spikes_df) == 0:
            return None
        
        weekly = self.analyze_spike_trends(spikes_df)
        if weekly is None or len(weekly) < 2:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # iAUC trend
        ax1 = axes[0, 0]
        ax1.plot(weekly['week'], weekly['mean_iauc'], marker='o', linewidth=2, markersize=8)
        ax1.fill_between(weekly['week'], 
                         weekly['mean_iauc'] - weekly['std_iauc'],
                         weekly['mean_iauc'] + weekly['std_iauc'],
                         alpha=0.3)
        ax1.set_xlabel('Week Number')
        ax1.set_ylabel('Mean iAUC (mmol/L·h)')
        ax1.set_title('Glucose Spike iAUC Over Time\n(Lower = Better Control)')
        ax1.grid(True, alpha=0.3)
        
        # Descent rate trend (insulin sensitivity)
        ax2 = axes[0, 1]
        ax2.plot(weekly['week'], weekly['mean_descent_rate'], 
                marker='o', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Week Number')
        ax2.set_ylabel('Mean Descent Rate (mmol/L/min)')
        ax2.set_title('Glucose Clearance Rate Over Time\n(Higher = Better Insulin Sensitivity)')
        ax2.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='Good threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Recovery time trend
        ax3 = axes[1, 0]
        ax3.plot(weekly['week'], weekly['mean_recovery_time'], 
                marker='o', linewidth=2, markersize=8, color='orange')
        ax3.set_xlabel('Week Number')
        ax3.set_ylabel('Mean Recovery Time (min)')
        ax3.set_title('Time to Return to Baseline\n(Lower = Better Clearance)')
        ax3.axhline(90, color='red', linestyle='--', alpha=0.5, label='Good threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Spike frequency
        ax4 = axes[1, 1]
        ax4.bar(weekly['week'], weekly['n_spikes'], alpha=0.7, color='purple')
        ax4.set_xlabel('Week Number')
        ax4.set_ylabel('Number of Spikes')
        ax4.set_title('Spike Frequency Over Time')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_distribution(self, figsize=(12, 5)):
        """Plot glucose distribution histogram"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(self.df['glucose'], bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(3.9, color='red', linestyle='--', label='Hypoglycemia')
        ax1.axvline(7.8, color='green', linestyle='--', label='Target')
        ax1.axvline(10.0, color='orange', linestyle='--', label='Hyperglycemia')
        ax1.set_xlabel('Glucose (mmol/L)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Glucose Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by day
        self.df['date'] = self.df['datetime'].dt.date
        daily = self.df.groupby('date')['glucose'].apply(list)
        
        if len(daily) <= 30:  # Only if reasonable number of days
            ax2.boxplot(daily.values, labels=range(1, len(daily)+1))
            ax2.set_xlabel('Day')
            ax2.set_ylabel('Glucose (mmol/L)')
            ax2.set_title('Daily Glucose Distribution')
            ax2.axhline(3.9, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(7.8, color='green', linestyle='--', alpha=0.5)
            ax2.axhline(10.0, color='orange', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Too many days to display\nindividual box plots', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        return fig


def main():
    """Main analysis pipeline"""
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python cgm_analysis.py <path_to_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Set output directory - use current directory for local runs
    output_dir = os.path.join(os.getcwd(), 'cgm_outputs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    print("=" * 60)
    print("CGM ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Initialize analyzer
    cgm = CGMAnalyzer(csv_path)
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    # Calculate summary stats
    stats = cgm.calculate_summary_stats()
    for key, value in stats.items():
        print(f"{key:.<40} {value:.2f}")
    
    # MAGE
    mage = cgm.calculate_mage()
    print(f"{'MAGE (mmol/L)':.<40} {mage:.2f}")
    
    # Fasting glucose
    print("\n" + "=" * 60)
    print("FASTING GLUCOSE (6-8 AM)")
    print("=" * 60)
    fasting = cgm.analyze_fasting_glucose()
    if fasting:
        for key, value in fasting.items():
            print(f"{key:.<40} {value:.2f}" if isinstance(value, float) else f"{key:.<40} {value}")
    else:
        print("No fasting glucose data available")
    
    # Detect spikes
    print("\n" + "=" * 60)
    print("GLUCOSE SPIKE DETECTION")
    print("=" * 60)
    print("Detecting post-prandial glucose spikes...")
    print("Parameters:")
    print("  - Baseline window: 30 minutes")
    print("  - Spike threshold: 1.5 mmol/L above baseline")
    print("  - Minimum spike duration: 15 minutes")
    
    spikes = cgm.detect_spikes(
        baseline_window=30, 
        spike_threshold=1.5, 
        min_spike_duration=15,
        min_peak_duration=10,
        return_to_baseline_threshold=0.5
    )
    
    print(f"\nNumber of spikes detected: {len(spikes)}")
    
    if len(spikes) > 0:
        print("\n" + "-" * 60)
        print("SPIKE SUMMARY STATISTICS")
        print("-" * 60)
        print(f"{'Mean iAUC (mmol/L·h)':.<45} {spikes['iauc'].mean():.2f} ± {spikes['iauc'].std():.2f}")
        print(f"{'Median iAUC (mmol/L·h)':.<45} {spikes['iauc'].median():.2f}")
        print(f"{'Mean Duration (min)':.<45} {spikes['duration_min'].mean():.1f} ± {spikes['duration_min'].std():.1f}")
        print(f"{'Mean Time to Peak (min)':.<45} {spikes['time_to_peak_min'].mean():.1f} ± {spikes['time_to_peak_min'].std():.1f}")
        print(f"{'Mean Recovery Time (min)':.<45} {spikes['recovery_time_min'].mean():.1f} ± {spikes['recovery_time_min'].std():.1f}")
        print(f"{'Mean Peak Glucose (mmol/L)':.<45} {spikes['peak_glucose'].mean():.2f} ± {spikes['peak_glucose'].std():.2f}")
        print(f"{'Mean Delta (mmol/L)':.<45} {spikes['delta'].mean():.2f} ± {spikes['delta'].std():.2f}")
        print(f"{'Mean Descent Rate (mmol/L/min)':.<45} {spikes['descent_rate'].mean():.3f} ± {spikes['descent_rate'].std():.3f}")
        print(f"{'Mean Clearance Efficiency (mmol/L/min)':.<45} {spikes['clearance_efficiency'].mean():.3f}")
        
        print("\n" + "-" * 60)
        print("INSULIN SENSITIVITY INDICATORS")
        print("-" * 60)
        print("Lower iAUC = Better glucose control")
        print("Faster descent rate = Better insulin sensitivity")
        print("Shorter recovery time = Better glucose clearance")
        print(f"\n{'Your Mean Descent Rate':.<45} {spikes['descent_rate'].mean():.3f} mmol/L/min")
        print(f"{'Your Mean Recovery Time':.<45} {spikes['recovery_time_min'].mean():.1f} minutes")
        print("\nBenchmark (general population):")
        print("  Good insulin sensitivity: descent rate >0.05 mmol/L/min")
        print("  Good glucose clearance: recovery time <90 minutes")
        
        print("\n" + "-" * 60)
        print("TOP 5 LARGEST SPIKES (by iAUC)")
        print("-" * 60)
        top_spikes = spikes.nlargest(5, 'iauc')[
            ['start_time', 'peak_glucose', 'delta', 'iauc', 'duration_min', 'recovery_time_min', 'descent_rate']
        ]
        print(top_spikes.to_string(index=False))
        
        print("\n" + "-" * 60)
        print("SLOWEST GLUCOSE CLEARANCE (longest recovery)")
        print("-" * 60)
        slow_spikes = spikes.nlargest(5, 'recovery_time_min')[
            ['start_time', 'peak_glucose', 'recovery_time_min', 'descent_rate', 'iauc']
        ]
        print(slow_spikes.to_string(index=False))
        
        # Save spike details
        spike_output = os.path.join(output_dir, 'glucose_spikes.csv')
        spikes.to_csv(spike_output, index=False)
        print(f"\n✓ Detailed spike data saved to: {spike_output}")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    fig1 = cgm.plot_glucose_trace()
    fig1_path = os.path.join(output_dir, 'glucose_trace.png')
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"✓ {fig1_path}")
    
    fig2 = cgm.plot_daily_patterns()
    fig2_path = os.path.join(output_dir, 'daily_patterns.png')
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"✓ {fig2_path}")
    
    fig3 = cgm.plot_distribution()
    fig3_path = os.path.join(output_dir, 'glucose_distribution.png')
    fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"✓ {fig3_path}")
    
    if len(spikes) > 0:
        fig4 = cgm.plot_spikes(spikes, max_spikes=10)
        if fig4:
            fig4_path = os.path.join(output_dir, 'individual_spikes.png')
            fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
            print(f"✓ {fig4_path}")
        
        fig5 = cgm.plot_spike_trends(spikes)
        if fig5:
            fig5_path = os.path.join(output_dir, 'spike_trends.png')
            fig5.savefig(fig5_path, dpi=300, bbox_inches='tight')
            print(f"✓ {fig5_path}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
