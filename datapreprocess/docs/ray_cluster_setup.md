# Readme: Using Instance Profile ARNs in Ray Cluster Configurations

## Overview

Instance Profile Amazon Resource Names (ARNs) are a secure method to grant an AWS EC2 instance the necessary permissions to access AWS resources. This is especially relevant in the context of Ray clusters, where managing resources such as EC2 instances and S3 buckets is common.

## Instance Profile ARNs in Ray Clusters

In Ray cluster configurations, you can specify an Instance Profile ARN for your EC2 instances. This allows the instances within the Ray cluster to inherit permissions from the IAM role associated with the specified Instance Profile, enabling them to interact with other AWS services like S3 and EC2 without the need to explicitly provide AWS credentials.

### Example Configuration

In the provided Ray cluster configuration, the `IamInstanceProfile` field is used:

```yaml
IamInstanceProfile:
    Arn: arn:aws:iam::[aws-account-id]:instance-profile/ray-autoscaler-v1
```

This configuration attaches the `ray-autoscaler-v1` instance profile to the EC2 instances in the cluster, granting them the permissions associated with that profile.

## Benefits

- **Security**: Using an Instance Profile ARN is more secure than storing AWS keys in scripts or configuration files.
- **Ease of Management**: Permissions can be managed centrally in the IAM role, without the need to update each instance or script.
- **Automatic Credential Rotation**: AWS handles the rotation of credentials, reducing the risk of using outdated or compromised keys.

## Anti-Pattern: Storing AWS Keys

It is a very bad practice to hardcode AWS keys in any script or configuration file. This exposes the risk of key leakage and makes it difficult to manage and rotate keys securely. Always use IAM roles and Instance Profile ARNs wherever possible.

## Setting Up in Ray Cluster Config

1. **Create an IAM Role**: First, create an IAM role with the necessary permissions for your Ray cluster.
2. **Create an Instance Profile**: Attach this IAM role to an instance profile.
3. **Update Ray Config**: Use the ARN of the created instance profile in your Ray cluster configuration under `IamInstanceProfile`.

## Conclusion

Leveraging Instance Profile ARNs in your Ray cluster configurations enhances security and simplifies permission management. It is crucial to avoid the anti-pattern of hardcoding AWS credentials in scripts or files for security and operational efficiency.
