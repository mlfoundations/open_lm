# Complete Guide to Setting Up a Ray Cluster with AWS Instance Profile

This guide provides a comprehensive walkthrough on setting up a Ray cluster configuration with an AWS Instance Profile, including steps to create the Instance Profile if it doesn't exist already.

## Part 1: Setting Up Ray Cluster Configuration

### Step 1: Basic Configuration
Start by defining the basic parameters of your Ray cluster in the configuration file:

```yaml
cluster_name: ray-shuffle-tokenize
max_workers: 25
upscaling_speed: 0.0
provider:
    type: aws
    region: us-west-2
    cache_stopped_nodes: False
```

### Step 2: Node Configuration
Configure the node types, specifying the instance types, image IDs, and most importantly, the IAM Instance Profile ARN:

```yaml
available_node_types:
    ray.head.default:
        resources: {}
        node_config:
            SubnetIds: [subnet-xxx, subnet-yyy, subnet-zzz]
            ImageId: ami-xxxxxxx # Example AMI ID
            InstanceType: i4i.8xlarge
            IamInstanceProfile:
                Arn: [Your-Instance-Profile-ARN]
    ray.worker.default:
        min_workers: 25
        max_workers: 25
        node_config:
            SubnetIds: [subnet-xxx, subnet-yyy, subnet-zzz]
            ImageId: ami-xxxxxxx # Example AMI ID
            InstanceType: i4i.8xlarge
            IamInstanceProfile:
                Arn: [Your-Instance-Profile-ARN]
```
Replace `[Your-Instance-Profile-ARN]` with the actual ARN of your instance profile. 

### Step 3: Setup Commands
Define any setup commands necessary for your environment:

```yaml
setup_commands:
    - wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O miniconda.sh
    # ... other setup commands ...
```

### Step 4: Security Best Practices
**Important**: Avoid hardcoding AWS credentials in your scripts or files. Using an IAM role through an Instance Profile is a more secure and recommended approach.

## Part 2: Creating an AWS Instance Profile (If Not Existing)

### Step 1: Create an IAM Role
1. **Open IAM in AWS Console**: Log into the AWS Management Console and navigate to the IAM (Identity and Access Management) service.
2. **Create a New Role**: Go to "Roles" > "Create role".
3. **Select EC2 as the Trust Entity**: Choose "AWS service" for the type of trusted entity and select "EC2".
4. **Attach Permissions**: Select `AmazonEC2FullAccess` and `AmazonS3FullAccess` policies for comprehensive EC2 and S3 access.
5. **Name and Create the Role**: Provide a name (e.g., `RayClusterRole`) and create the role.

### Step 2: Create the Instance Profile
1. **Navigate to the Role**: In IAM roles, find the newly created role.
2. **Create Instance Profile**: Under the "Role actions" menu, select "Add role to instance profile".
3. **Name the Instance Profile**: Give the instance profile the same name as the role for consistency.

### Step 3: Retrieve the Instance Profile ARN
1. **Open the Role Details**: Click on the role you just created.
2. **Copy the Instance Profile ARN**: In the summary section, you'll find the ARN which looks like `arn:aws:iam::[aws-account-id]:instance-profile/RayClusterRole`.

### Step 4: Update Ray Cluster Config
Replace `[Your-Instance-Profile-ARN]` in your Ray cluster configuration with the ARN you just copied.

