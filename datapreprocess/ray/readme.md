Go into cluster_test.yaml and enter AWS credentials

Spin up ray cluster with ray up cluster_test.yaml

Access ray cluster with ray attach cluster_test.yaml

Send tokenize_shuffle.py to cluster with ray rsync_up cluster_test.yaml tokenize_shuffle.py /home/ubuntu

To just tokenize and break into context lengths, run: python tokenize_shuffle.py --input “s3://dcnlp-data/redpajamas-raw/c4-train.{00000..00063}-of-01024.jsonl” --output s3://dcnlp-data/tokenize-shuffle-test/ --no_shuffle

To run the above and then also shuffle, run: python tokenize_shuffle.py --input “s3://dcnlp-data/redpajamas-raw/c4-train.{00000..00063}-of-01024.jsonl” --output s3://dcnlp-data/tokenize-shuffle-test/

Make sure that the above paths are in the same region as the region specified in the ray yaml file (currently us-east-1)

Can exit cluster normally and reenter when done.

IMPORTANT: When done, ray down cluster_test.yaml - if you do not do this, the instances will stay up and incur additional costs

IMPORTANT: DO NOT push AWS credentials to the git repo
