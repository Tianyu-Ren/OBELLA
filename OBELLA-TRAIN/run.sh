#!/bin/bash
encoder_state="albert/albert-xxlarge-v2"
num_class=3
batch_size=128
test_batch_size=64
max_length=256
epoch=2
num_steps=-1
lr=1e-5
warmup_ratio=0.1
accumulation_steps=1
seed=1


python train.py --encoder_state $encoder_state --num_class $num_class --batch_size $batch_size --test_batch_size $test_batch_size \
--max_length $max_length --epoch $epoch --num_steps $num_steps --lr $lr --warmup_ratio $warmup_ratio  --seed $seed  --accumulation_steps $accumulation_steps
