# conda-env: mtech-env
import pandas as pd
import numpy as np
import os
import glob
import time
import warnings
from tqdm import tqdm
from dowhy import gcm
from dowhy.gcm.util.general import set_random_seed

# Set seed for reproducibility
set_random_seed(42)

class CounterfactualAnalyzer:
    def __init__(self, causal_model, original_dir="original", cf_dir="counterfactuals", output_dir="worked"):
        """Initialize the counterfactual analyzer with directories and model."""
        self.causal_model = causal_model
        self.original_dir = original_dir
        self.cf_dir = cf_dir
        self.output_dir = output_dir
        self.distance_features = ['cp', 'trestbps', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.all_counterfactuals = []
        self.processed_instances = []
        self.stats = {
            'total_cf': 0,
            'working_cf': 0,
            'failed_instances': 0,
            'processing_time': 0
        }
    
    def load_instance(self, instance_num):
        """Load original and counterfactual data for a given instance."""
        original_file = f"{self.original_dir}/{instance_num}_original.csv"
        cf_file = f"{self.cf_dir}/{instance_num}_counterfactual.csv"
        
        # Check if files exist
        if not os.path.exists(original_file) or not os.path.exists(cf_file):
            return None, None
        
        # Load original data and add target
        orig_data = pd.read_csv(original_file)
        orig_data = orig_data.assign(target=1)
        
        # Load counterfactual data
        cf_data = pd.read_csv(cf_file)
        
        return orig_data, cf_data
    
    def generate_counterfactuals(self, instance_num, orig_data, cf_data):
        """Generate counterfactuals for a given instance using interventions."""
        results = []
        original_values = orig_data.iloc[0].to_dict()
        
        for cf_idx in range(len(cf_data)):
            # Determine intervention type and value
            cf_chol = cf_data['chol'].iloc[cf_idx] if 'chol' in cf_data.columns else None
            cf_trestbps = cf_data['trestbps'].iloc[cf_idx] if 'trestbps' in cf_data.columns else None
            
            # Skip if both interventions are invalid
            if pd.isna(cf_chol) and pd.isna(cf_trestbps):
                continue
                
            # Create intervention dictionary
            if pd.notna(cf_chol):
                intervention_dict = {'chol': lambda chol: cf_chol}
                intervention_type = "chol"
                intervention_value = cf_chol
            else:
                intervention_dict = {'trestbps': lambda trestbps: cf_trestbps}
                intervention_type = "trestbps"
                intervention_value = cf_trestbps
            
            try:
                # Generate counterfactual samples
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cf_samples = gcm.interventional_samples(self.causal_model, 
                                                           intervention_dict, 
                                                           observed_data=orig_data)
                
                # Apply constraints to keep outputs physiologically valid
                cf_samples['oldpeak'] = cf_samples['oldpeak'].clip(lower=0)
                cf_samples['exang'] = cf_samples['exang'].clip(lower=0, upper=1)
                cf_samples['cp'] = cf_samples['cp'].round().clip(lower=1, upper=4)
                cf_samples['slope'] = cf_samples['slope'].round().clip(lower=1, upper=3)
                cf_samples['restecg'] = cf_samples['restecg'].round().clip(lower=0, upper=2)
                
                # Process each counterfactual sample
                for _, row in cf_samples.iterrows():
                    cf_result = self.create_comparison_row(
                        instance_num, cf_idx, intervention_type, intervention_value,
                        original_values, row
                    )
                    results.append(cf_result)
                    
            except Exception as e:
                print(f"Error processing instance {instance_num}, CF #{cf_idx}: {str(e)}")
        
        return results
    
    def create_comparison_row(self, instance_num, cf_idx, intervention_type, 
                             intervention_value, original_values, cf_values):
        """Create a comparison row with original and CF values, distances, etc."""
        comparison = {
            'instance_id': instance_num,
            'cf_id': cf_idx,
            'intervention_type': intervention_type,
            'intervention_value': intervention_value,
        }
        
        # Add original and counterfactual values
        for col in original_values.keys():
            comparison[f'orig_{col}'] = original_values[col]
        for col in cf_values.index:
            comparison[f'cf_{col}'] = cf_values[col]
        
        # Calculate distances and changes
        euclidean_distance = 0
        for feature in self.distance_features:
            # Get values as floats
            orig_val = float(original_values[feature])
            cf_val = float(cf_values[feature])
            
            # Calculate differences
            abs_diff = abs(orig_val - cf_val)
            signed_diff = cf_val - orig_val
            
            # Store metrics
            comparison[f'dist_{feature}'] = abs_diff
            comparison[f'change_{feature}'] = signed_diff
            
            # Calculate percentage change if original value isn't zero
            if orig_val != 0:
                pct_change = (signed_diff / orig_val) * 100
                comparison[f'pct_change_{feature}'] = pct_change
            else:
                comparison[f'pct_change_{feature}'] = np.nan
                
            # Accumulate for overall distance
            euclidean_distance += abs_diff**2
            
        # Overall distance metrics
        comparison['euclidean_distance'] = np.sqrt(euclidean_distance)
        comparison['manhattan_distance'] = sum(comparison[f'dist_{feature}'] 
                                            for feature in self.distance_features)
            
        # Target change indicator
        comparison['target_changed'] = 1 if (original_values['target'] == 1 and 
                                           cf_values['target'] == 0) else 0
            
        return comparison
    
    def process_instance(self, instance_num):
        """Process a single instance and return its counterfactuals."""
        orig_data, cf_data = self.load_instance(instance_num)
        
        if orig_data is None or cf_data is None:
            return []
        
        results = self.generate_counterfactuals(instance_num, orig_data, cf_data)
        
        # Save individual instance results
        if results:
            df = pd.DataFrame(results)
            df.to_csv(f"{self.output_dir}/{instance_num}_all_counterfactuals.csv", index=False)
            
            # Keep track of processed instances
            self.processed_instances.append(instance_num)
            
        return results
    
    def process_all_instances(self, instance_range=range(48), show_progress=True):
        """Process all instances in the given range."""
        start_time = time.time()
        
        # Create progress bar if requested
        instances = list(instance_range)
        if show_progress:
            instances = tqdm(instances, desc="Processing instances")
        
        # Process each instance
        for instance_num in instances:
            print(f"\nProcessing instance {instance_num}")
            results = self.process_instance(instance_num)
            
            if results:
                self.all_counterfactuals.extend(results)
                self.stats['total_cf'] += len(results)
                self.stats['working_cf'] += sum(r['target_changed'] for r in results)
            else:
                self.stats['failed_instances'] += 1
        
        # Store overall processing time
        self.stats['processing_time'] = time.time() - start_time
        
        # Save combined results
        self.save_results()
        
        # Display summary
        self.print_summary()
    
    def save_results(self):
        """Save all results to CSV files."""
        if self.all_counterfactuals:
            # Convert to DataFrame
            all_cf_df = pd.DataFrame(self.all_counterfactuals)
            
            # Save all counterfactuals
            all_cf_df.to_csv(f"{self.output_dir}/all_counterfactuals_with_distances.csv", index=False)
            
            # Save working counterfactuals separately
            worked_cf = all_cf_df[all_cf_df['target_changed'] == 1]
            worked_cf.to_csv(f"{self.output_dir}/working_counterfactuals_with_distances.csv", index=False)
    
    def print_summary(self):
        """Print a summary of the processing results."""
        print("\n" + "="*50)
        print("COUNTERFACTUAL ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total instances processed: {len(self.processed_instances)}")
        print(f"Instances with counterfactuals: {len(self.processed_instances)}")
        print(f"Failed instances: {self.stats['failed_instances']}")
        print(f"Total counterfactuals generated: {self.stats['total_cf']}")
        print(f"Working counterfactuals (target=0): {self.stats['working_cf']}")
        print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        print("="*50)
        
        if self.all_counterfactuals:
            df = pd.DataFrame(self.all_counterfactuals)
            
            print("\nDistance metrics for all counterfactuals:")
            print(df[['euclidean_distance', 'manhattan_distance']].describe())
            
            worked_df = df[df['target_changed'] == 1]
            if not worked_df.empty:
                print("\nDistance metrics for working counterfactuals:")
                print(worked_df[['euclidean_distance', 'manhattan_distance']].describe())
                
                # Feature change analysis for working counterfactuals
                print("\nFeature changes in working counterfactuals:")
                change_cols = [col for col in worked_df.columns if col.startswith('change_')]
                print(worked_df[change_cols].describe())

