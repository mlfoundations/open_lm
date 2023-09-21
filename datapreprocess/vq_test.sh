#!/bin/bash

python3 make_vq.py \
	--input-files "CVE_tokens" \
	--output-dir "tokens" \
	--num-workers 64 \
	--num-consumers 128 \
	--upload-to-s3 \
