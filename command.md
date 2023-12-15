```bash
torchrun --nproc_per_node=2 train.py --data-path /root/autodl-tmp/label_gen/dataset -b 16 -j 8 --opt adamw --lr 0.01 --label-smoothing 0.11 --mixup-alpha 0.2  --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 3 --lr-warmup-decay 0.033 --ra-sampler --amp --model-ema --train-crop-size 544 --val-resize-size 544 --clip-grad-norm 1 --output-dir output_resnext

torchrun --nproc_per_node=2 train.py --data-path /root/autodl-tmp/label_gen/dataset -b 16 -j 8 --opt adamw --lr 0.004 --label-smoothing 0.11 --mixup-alpha 0.2  --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 3 --lr-warmup-decay 0.033  --auto-augment ta_wide  --ra-sampler --ra-reps 4 --amp --model-ema --train-crop-size 544 --val-resize-size 544 --train-crop-size 544 --val-crop-size 544 --clip-grad-norm 1 --output-dir output_convnext_large

```