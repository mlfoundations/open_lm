# concurrent eval script
# must change all paths to be your own
# example usage
# bash experiments/base-mw/eval.sh 200b-rpj-neox-g3b_neox-10-1e-3-0.1-nodes48-bs10-v0 g3b_neox

OPEN_CLIP_HOME="/fsx/home-suching/open_lm"
export PYTHONPATH="$PYTHONPATH:${OPEN_CLIP_HOME}"


cd /fsx/home-suching/open_lm/


folder=$1
etype=$2

while true
do
    for i in `ls -t /fsx/home-suching/experiments/mix_wo/$folder/checkpoints/epoch*.pt`
    do

        #echo $model
        save_path="$(dirname $i)/val_$(basename $i)"

        echo $save_path


        if [ -f "$save_path" ]; then
            echo "$save_path exists."
        elif [[ $save_path == *"latest"* ]]; then
            echo "pass on latest"
        else
            torchrun --nproc-per-node 8 -m open_lm.main \
                --val-data "pipe:aws s3 cp s3://laion-west/lmdata/validation_data_tokenized/open_lm//shard_00000000.tar -" \
                --workers 6 \
                --precision amp_bfloat16 \
                --batch-size 8 \
                --grad-checkpointing \
                --log-every-n-steps 1 \
                --model $etype \
                --fsdp --fsdp-amp \
                --data-key json \
                --train-num-samples 1000000000 \
                --model-norm gain_only_layer_norm \
                --name $RANDOM \
                --resume "$i" \
                --logs /fsx/home-suching/experiments/lmdebug  > $save_path
        fi
    done
    sleep 60
done
