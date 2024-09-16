import re
import pandas as pd
import argparse

def extract_metrics(log_file, output_csv):
    # Define regex pattern for extracting metrics
    pattern = r"Epoch (\d+)/\d+ metrics - D2d: ([\d.]+), P2d: ([\d.]+), MPJPE: ([\d.]+), PVE: ([\d.]+), PA-MPJPE: ([\d.]+), PA-PVE: ([\d.]+)"
    
    # Initialize a list to store the metrics data
    metrics_data = []
    
    # Open and read the log file
    with open(log_file, 'r') as file:
        log_data = file.read()
        
        # Extract all metrics lines using regex
        metrics_data = re.findall(pattern, log_data)
        
    # Define the column headers
    columns = ['Epoch', 'D2d', 'P2d', 'MPJPE', 'PVE', 'PA-MPJPE', 'PA-PVE']
    
    # Convert the extracted data to a pandas DataFrame
    df = pd.DataFrame(metrics_data, columns=columns)
    
    # Convert the necessary columns to floats
    df[['D2d', 'P2d', 'MPJPE', 'PVE', 'PA-MPJPE', 'PA-PVE']] = df[['D2d', 'P2d', 'MPJPE', 'PVE', 'PA-MPJPE', 'PA-PVE']].astype(float)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Metrics saved to {output_csv}")

def main():
    # Create an argument parser to take the log file and output path as inputs
    parser = argparse.ArgumentParser(description='Extract metrics from a training log file and save to CSV.')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the CSV file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function to extract metrics and save to CSV
    extract_metrics(args.log_file, args.output_csv)

if __name__ == "__main__":
    main()