
# Ray Cluster Setup and Execution Guide

## Quick Commands

1. **Spin up the ray cluster**:  
   ```
   ray up ray_cluster_configs/cluster_west.yaml
   ```

2. **Access the ray cluster**:  
   ```
   ray attach ray_cluster_configs/cluster_west.yaml
   ```

3. **Transfer the `tokenize_shuffle.py` script to the cluster**:  
   ```
   ray rsync_up ray_cluster_configs/cluster_west.yaml tokenize_shuffle.py /home/ubuntu
   ```

5. **Tokenize with shuffling**:  
   ```
   python tokenize_shuffle.py --input “s3://dcnlp-data/redpajamas-raw/c4-train.{00000..00063}-of-01024.jsonl” --output s3://dcnlp-data/tokenize-shuffle-test/
   ```

> **Note**: Ensure that the paths specified above are in the same AWS region as the one mentioned in the ray yaml file (currently set to `us-west-2`).

6. **Exit and re-enter the cluster as required**.

## Detailed Workflow

1. **Configure AWS**:  
   Start by setting up your AWS credentials:
   ```
   aws configure
   ```

2. **Initialize the cluster**:  
   ```
   ray up ray_cluster_configs/cluster_west.yaml
   ```

3. **Copy the script to the cluster**:  
   ```
   ray rsync_up ray_cluster_configs/cluster_west.yaml tokenize_shuffle.py /home/ubuntu
   ```
   Copy the `default_dataset_yaml` as well if used.

4. **SSH into the cluster**:  
   ```
   ray attach ray_cluster_configs/cluster_west.yaml
   ```

5. **Enter tmux and execute the job**:  
   ```
   tmux new-session -d -s ray_tokenize_shuffle  'python tokenize_shuffle.py'
   ```

> **Heads up**: This is version 0 of this script. The user interface will be improved in future versions. Currently, objects are being spilled to `dcnlp-hub`.
