import os
import sys
import subprocess
import time

# Define the number of times you want to run the code
num_runs = 1

# Create a directory to store the output files
output_dir = "./output/"
os.makedirs(output_dir, exist_ok=True)

# Define the command or script you want to run
command = "python my_experiment.py --is_onehot True --save_model True --log_to_wandb True"

# Run the command multiple times
for i in range(num_runs):
    # Generate a unique filename for each run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f"{output_dir}/output_{timestamp}.txt"

    # Redirect the terminal output to the output file
    with open(output_file, "w") as f:
        subprocess.run(command+f" --save_path ./output/{timestamp}.pth --png_path ./output/{timestamp}", shell=True, stdout=f, stderr=subprocess.STDOUT)

    print(f"Run {i+1}/{num_runs} completed. Output saved to {output_file}")

print("All runs completed.")