sources=(
    mbm_paper_eval
    mbm_paper_eval_2
    mbm_paper_eval_3
    mbm_paper_eval_4
    mbm_paper_eval_5
    mbm_paper_eval_6
    mbm_paper_eval_1b
)
for source in "${sources[@]}"; do
    mkdir -p results/$source/aggregated/
    aws s3 sync s3://tri-ml-datasets/openlm/$source/results/aggregated/ results/$source/aggregated/
    AWS_DEFAULT_REGION=us-west-2 aws s3 sync s3://tri-ml-datasets-uw2/openlm/$source/results/aggregated/ results/$source/aggregated/ --profile=poweruser
done

