
python scripts/copy_results.py --bucket 1b
aws s3 sync results/mbm_paper_texteval_1b/ s3://tri-ml-datasets/mbm/exp_data/eval/text_only/mbm_paper_texteval_1b/

python scripts/copy_results.py --bucket 1
aws s3 sync results/mbm_paper_texteval_1/ s3://tri-ml-datasets/mbm/exp_data/eval/text_only/mbm_paper_texteval_1/

python scripts/copy_results.py --bucket 2
aws s3 sync results/mbm_paper_texteval_2/ s3://tri-ml-datasets/mbm/exp_data/eval/text_only/mbm_paper_texteval_2/

python scripts/copy_results.py --bucket 3
aws s3 sync results/mbm_paper_texteval_3/ s3://tri-ml-datasets/mbm/exp_data/eval/text_only/mbm_paper_texteval_3/

python scripts/copy_results.py --bucket 4
aws s3 sync results/mbm_paper_texteval_4/ s3://tri-ml-datasets/mbm/exp_data/eval/text_only/mbm_paper_texteval_4/

python scripts/copy_results.py --bucket 5 
aws s3 sync results/mbm_paper_texteval_5/ s3://tri-ml-datasets/mbm/exp_data/eval/text_only/mbm_paper_texteval_5/

python scripts/copy_results.py --bucket 6
aws s3 sync results/mbm_paper_texteval_6/ s3://tri-ml-datasets/mbm/exp_data/eval/text_only/mbm_paper_texteval_6/