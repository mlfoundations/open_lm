import argparse
import time
import os
import subprocess
import yaml
from datetime import datetime

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker_ssh_helper.wrapper import SSHEstimatorWrapper
from sagemaker_train.sm_utils import get_arn, get_remote_sync


NAME = "openlm"
INSTANCE_MAPPER = {
    "p4": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
}


def run_command(command):
    subprocess.run(command, shell=True, check=True)


def get_image(user, region, instance_type, build_image=False, update_image=False, profile="poweruser"):
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    algorithm_name = f"{user}-{NAME}"
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"
    if not build_image and not update_image:
        return fullname

    login_cmd = f"aws ecr get-login-password --region {region} --profile {profile} | docker login --username AWS --password-stdin"

    if build_image:
        print("Building container")
        commands = [
            f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
            f"docker build -f sagemaker_train/Dockerfile --build-arg AWS_REGION={region} --build-arg DOCKER_IGNORE_FILE=sagemaker_train/.dockerignore -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
            f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} || aws --region {region} ecr create-repository --repository-name {algorithm_name}",
        ]
    elif update_image:
        print("Updating container")
        commands = [
            f"docker build -f sagemaker_train/update.dockerfile --build-arg DOCKER_IGNORE_FILE=sagemaker_train/.dockerignore --build-arg BASE_DOCKER={algorithm_name} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
        ]

    print("\n".join(commands))
    subprocess.run("\n".join(commands), shell=True)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)

    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build image from scratch")
    parser.add_argument("--update", action="store_true", help="Update code in image, don't re-build")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--user", required=True, help="User name")
    parser.add_argument("--cfg-path", required=True, help="Launch config")

    # AWS profile args
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", default="poweruser", help="AWS profile to use")
    parser.add_argument("--arn", default=None, help="If None, reads from SAGEMAKER_ARN env var")
    parser.add_argument(
        "--s3-remote-sync", default=None, help="S3 path to sync to. If none, reads from S3_REMOTE_SYNC env var"
    )

    # Instance args
    parser.add_argument("--instance-count", default=1, type=int, help="Number of instances")
    parser.add_argument("--instance-type", default="p4de", choices=list(INSTANCE_MAPPER.keys()))
    parser.add_argument("--spot-instance", action="store_true")
    args = parser.parse_args()

    setup_tmp_name = "./setup_renamed_for_sagemaker.py"
    # print(f"Renaming ./setup.py to {setup_tmp_name}")
    # os.rename("./setup.py", setup_tmp_name)
    try:
        main_after_setup_move(args)
    except:
        # os.rename(setup_tmp_name, "./setup.py")
        raise


def main_after_setup_move(args):
    image = get_image(
        args.user,
        args.region,
        args.instance_type,
        build_image=args.build,
        update_image=args.update,
        profile=args.profile,
    )

    ##########
    # Create session and make sure of account and region
    ##########
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=args.region))

    # provide a pre-existing role ARN as an alternative to creating a new role
    role = get_arn(args.arn)
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

    with open(args.cfg_path, "r") as f:
        train_args = yaml.safe_load(f)
    train_args["logs"] = checkpoint_local_path if not args.local else "./logs/debug"

    def get_job_name(base, train_args):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"

        job_name = "_".join([base, date_str])

        return job_name

    job_name = get_job_name(base_job_name, train_args)

    s3_remote_sync = get_remote_sync(args.s3_remote_sync)
    output_root = f"{s3_remote_sync}/sagemaker/{args.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)

    estimator = PyTorch(
        entry_point="open_lm/main.py",
        base_job_name=base_job_name,
        hyperparameters=train_args,
        role=role,
        image_uri=image,
        instance_count=int(args.instance_count),
        instance_type="local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type],
        train_use_spot_instances=True if args.spot_instance else False,
        # sagemaker_session=sagemaker_session,
        output_path=output_s3,
        job_name=job_name,
        checkpoint_s3_uri=None if args.local else f"{output_s3}/checkpoint",
        checkpoint_local_path=None if args.local else checkpoint_local_path,
        code_location=output_s3,
        # Training using SMDataParallel Distributed Training Framework
        distribution={"torch_distributed": {"enabled": True}},
        # Max run 10 days
        max_run=5 * 24 * 60 * 60,
        max_wait=5 * 24 * 60 * 60 if args.spot_instance else None,
        # max_run=60 * 60,  # 60 minutes
        input_mode="FastFile",
        # environment={"TORCH_DISTRIBUTED_DEBUG": "DETAIL", "TORCH_CPP_LOG_LEVEL": "INFO"},
        keep_alive_period_in_seconds=30 * 60 if not args.spot_instance else None,  # 30 minutes
        dependencies=[SSHEstimatorWrapper.dependency_dir()],
    )

    # ssh_wrapper = SSHEstimatorWrapper.create(estimator, connection_wait_time_seconds=600)
    # dataset_location = "s3://tri-ml-datasets/scratch/achal.dave/projects/lavis/data/"
    estimator.fit(
        # inputs={"datasets": TrainingInput(dataset_location, input_mode="FastFile")}
    )
    # print("Job name:", estimator.latest_training_job.name)
    # print(f"To connect over SSH run: sm-local-ssh-training connect {ssh_wrapper.training_job_name()}")

    # instance_ids = ssh_wrapper.get_instance_ids(timeout_in_sec=900)  # <--NEW--

    # print(f"To connect over SSM run: aws ssm start-session --target {instance_ids[0]}")
    # estimator.logs()


if __name__ == "__main__":
    main()
