import argparse
import time
import os
import subprocess
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


NAME = "openlm-main"
INSTANCE_MAPPER = {
    "p4": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
}


def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def get_image(user, instance_type, build_type=None, profile="default", region="us-east-1"):
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    docker_dir = Path(__file__).parent
    if instance_type in ("p4", "p4de"):
        algorithm_name = f"{user}-{NAME}-p4"
        dockerfile_base = docker_dir / "Dockerfile"
        dockerfile_update = docker_dir / "Dockerfile_update"
    elif instance_type == "p5":
        algorithm_name = f"{user}-{NAME}-p5"
        dockerfile_base = docker_dir / "Dockerfile"
        dockerfile_update = docker_dir / "Dockerfile_update"
    else:
        raise ValueError(f"Unknown instance_type: {instance_type}")
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"
    if build_type is None:
        return fullname

    login_cmd = f"aws ecr get-login-password --region {region} --profile {profile} | docker login --username AWS --password-stdin"

    if build_type == "full":
        print("Building container")
        commands = [
            # Log in to Sagemaker account to get image.
            f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
            f"docker build --progress=plain -f {dockerfile_base} --build-arg AWS_REGION={region} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
            (
                f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} || "
                f"aws --region {region} ecr create-repository --repository-name {algorithm_name}"
            ),
        ]
    elif build_type == "update":
        print("Updating container")
        commands = [
            f"docker build --progress=plain -f {dockerfile_update} --build-arg BASE_DOCKER={algorithm_name} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
        ]
    else:
        raise ValueError(f"Unknown build_type: {build_type}")

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-type", choices=["full", "update"], help="Build image from scratch")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--user", required=True, help="User name")
    parser.add_argument("--cfg-path", required=True, help="Launch config")

    # AWS profile args
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", default="default", help="AWS profile to use")
    parser.add_argument("--arn", default=None, help="If None, reads from SAGEMAKER_ARN env var")
    parser.add_argument(
        "--s3-remote-sync", default=None, help="S3 path to sync to. If none, reads from S3_REMOTE_SYNC env var"
    )

    # Instance args
    parser.add_argument("--instance-count", default=1, type=int, help="Number of instances")
    parser.add_argument("--instance-type", default="p4de", choices=list(INSTANCE_MAPPER.keys()))
    parser.add_argument("--spot-instance", action="store_true")

    args = parser.parse_args()
    main_after_setup_move(args)


def main_after_setup_move(args):
    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        args.arn = os.environ["SAGEMAKER_ARN"]

    if args.s3_remote_sync is None:
        assert (
            "S3_REMOTE_SYNC" in os.environ
        ), "Please specify --s3-remote-sync or set the S3_REMOTE_SYNC environment variable"
        args.s3_remote_sync = os.environ["S3_REMOTE_SYNC"]

    image = get_image(
        args.user,
        args.instance_type,
        region=args.region,
        build_type=args.build_type,
        profile=args.profile,
    )

    ##########
    # Create session and make sure of account and region
    ##########
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=args.region))

    if args.local:
        from sagemaker.local import LocalSession

        sagemaker_session = LocalSession()

    role = args.arn
    # provide a pre-existing role ARN as an alternative to creating a new role
    role_name = role.split(["/"][-1])
    print(f"SageMaker Execution Role:{role}")
    print(f"The name of the Execution role: {role_name[-1]}")

    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    print(f"AWS account:{account}")

    session = boto3.session.Session()
    region = session.region_name
    print(f"AWS region:{region}")

    ##########
    # Configure the training
    ##########
    base_job_name = f"{args.user.replace('.', '-')}-{NAME}"

    checkpoint_local_path = "/opt/ml/checkpoints"

    def get_job_name(base):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"

        job_name = "_".join([base, date_str])

        return job_name

    job_name = get_job_name(base_job_name)

    output_root = f"{args.s3_remote_sync}/sagemaker/{args.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)

    estimator = PyTorch(
        entry_point="open_lm/main.py",
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters={"config": args.cfg_path},
        role=role,
        image_uri=image,
        instance_count=args.instance_count,
        instance_type="local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type],
        train_use_spot_instances=args.spot_instance,
        output_path=output_s3,
        job_name=job_name,
        checkpoint_s3_uri=None if args.local else f"{output_s3}/checkpoint",
        checkpoint_local_path=None if args.local else checkpoint_local_path,
        code_location=output_s3,
        # Training using SMDataParallel Distributed Training Framework
        distribution={"torch_distributed": {"enabled": True}},
        # Max run 5 days
        max_run=5 * 24 * 60 * 60,
        max_wait=5 * 24 * 60 * 60 if args.spot_instance else None,
        input_mode="FastFile",
        # environment={"TORCH_DISTRIBUTED_DEBUG": "DETAIL", "TORCH_CPP_LOG_LEVEL": "INFO"},
        environment={"SM_USE_RESERVED_CAPACITY": "1"},
        keep_alive_period_in_seconds=30 * 60 if not args.spot_instance else None,  # 30 minutes
    )

    estimator.fit()


if __name__ == "__main__":
    main()
