
export PYTHONPATH=/home/jean/codes/lm-evaluation-harness:/home/jean/codes/mbm-prismatic-dev:$PYTHONPATH

# (
# echo "Running lm_eval for the 79m CC150 model"
# CUDA_VISIBLE_DEVICES=2 lm_eval --model open_lm \
#     --model_args config_file=checkpoints/79m_CCx150/params.txt,checkpoint=checkpoints/79m_CCx150/checkpoints/epoch_21.pt,pretrained=79m_cc150_epoch21,tokenizer=EleutherAI/gpt-neox-20b \
#     --tasks agieval_sat_en,arc_easy,bigbench_conceptual_combinations_multiple_choice,bigbench_cs_algorithms_multiple_choice,boolq,copa,hellaswag,mathqa,piqa,pubmedqa \
#     --output_path lm_eval_logs/79m_CCx150 \
#     --device cuda \
#     --log_samples \
#     --batch_size 1
# ) &

model_names=(

    ############################################### 8
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"

####################### 4

    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=5+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava-0p80_0p20-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=5+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p008_0p002_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=5+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"

############################################################################################################ 8

    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p04_0p01_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=5+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+datacompdr1b_caption_llava_dclm-0p08_0p02_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=5+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=1+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=2+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=3+stage-finetune+x7"
    "llava-multimodal+dclm-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=4+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=1+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=2+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=3+stage-finetune+x7"
    "llava-multimodal+llava-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=4+stage-finetune+x7"
    "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"

############################################################################################################ 10

    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p01_0p99-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p05_0p95-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0009788970261355868-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p0034466737184832607-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p006499974971416897-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=0-lr=0p009080349589287376-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1+epochs=4+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=1+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=2+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=3+stage-finetune+x7"
    # "llava-multimodal+llava_dclm-0p10_0p90-1b-fused-siglip-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1p0-seed=124-replacement=1-m=bucket6+epochs=4+stage-finetune+x7"
)
bucket_name=mbm_paper_finetune_1b
source_s3=s3://tri-ml-datasets/openlm
target_download=/datasets/jean/checkpoints/$bucket_name

# Function to get the number of available GPUs
get_gpu_count() {
    nvidia-smi --list-gpus | wc -l
}

# Function to get available memory on a GPU
get_gpu_memory() {
    local gpu=$1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu
}

# Initialize an array to store the last use time for each GPU
declare -A last_use_time
gpu_count=$(get_gpu_count)
for ((i=0; i<$gpu_count; i++)); do
    last_use_time[$i]=0
done

# Function to find the best available GPU
find_best_gpu() {
    local best_gpu=-1
    local max_memory=0
    local current_time=$(date +%s)
    
    for ((gpu=0; gpu<$gpu_count; gpu++)); do
        local available_memory=$(get_gpu_memory $gpu)
        local time_since_last_use=$((current_time - ${last_use_time[$gpu]}))
        
        # Check if GPU has more than 10GB available and hasn't been used in the last 60 seconds
        if [ $available_memory -gt 10240 ] && [ $time_since_last_use -gt 180 ]; then
            if [ $available_memory -gt $max_memory ]; then
                max_memory=$available_memory
                best_gpu=$gpu
            fi
        fi
    done
    
    echo $best_gpu
}

# Function to run task on a specific GPU
run_task() {
    local model_name=$1
    local gpu=$2

    mkdir -p $target_download/$model_name/checkpoints
    mkdir -p lm_eval_logs/$bucket_name/$model_name

    echo "Downloading model from $source_s3/$bucket_name/$model_name"
    aws s3 sync $source_s3/$bucket_name/$model_name/checkpoints/ $target_download/$model_name/checkpoints/ --exclude "*" --include "*latest-checkpoint.pt"
    
    echo "Downloading params"
    aws s3 cp $source_s3/$bucket_name/$model_name/config.json $target_download/$model_name/config.json
    
    echo "Running lm_eval for the $model_name model on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu lm_eval --model prismatic \
        --model_args pretrained="$target_download/$model_name",model_id="(openvlm-siglip)$target_download/$model_name",tokenizer=EleutherAI/gpt-neox-20b \
        --tasks agieval_sat_en,arc_easy,bigbench_conceptual_combinations_multiple_choice,bigbench_cs_algorithms_multiple_choice,boolq,copa,hellaswag,mathqa,piqa,pubmedqa \
        --device cuda \
	    --output_path lm_eval_logs/$bucket_name/$model_name \
        --batch_size 1
}

# Main execution
for model_name in "${model_names[@]}"; do
    while true; do
        gpu=$(find_best_gpu)
        if [ $gpu -ne -1 ]; then
            last_use_time[$gpu]=$(date +%s)
            run_task "$model_name" "$gpu" &
            break
        else
            sleep 10  # Wait for 10 seconds before checking again
        fi
    done
done

# Wait for all background tasks to complete
wait

# chatGPT above random:
# agieval_lsat_ar,agieval_lsat_rc,agieval_sat_en,arc_easy,bigbench_cs_algorithms_multiple_choice,bigbench_misconceptions_multiple_choice,bigbench_strange_stories_multiple_choice,bigbench_strategyqa_multiple_choice,copa,hellaswag,piqa,winogrande

# Handpicked above random:
#agieval_sat_en,arc_easy,bigbench_conceptual_combinations_multiple_choice,bigbench_cs_algorithms_multiple_choice,boolq,copa,hellaswag,mathqa,piqa,pubmedqa


# Initial tested tasks:
# agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,arc_challenge,arc_easy,bigbench_cs_algorithms_multiple_choice,bigbench_conceptual_combinations_multiple_choice,bigbench_dyck_languages_multiple_choice,bigbench_elementary_math_qa_multiple_choice,bigbench_language_identification_multiple_choice,bigbench_logical_deduction_multiple_choice,bigbench_misconceptions_multiple_choice,bigbench_strange_stories_multiple_choice,bigbench_strategyqa_multiple_choice,bigbench_understanding_fables_multiple_choice,boolq,copa,hellaswag,logiqa,logiqa2,mathqa,openbookqa,piqa,pubmedqa,winogrande
