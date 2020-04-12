this is the execution line for running the training
(set the eval-bleu-args for defult in translation.py because of argument issues
pass --eval-tokenized-bleu as well (otherwise bleu is not working)
pass max-epoch to decide the max epochs to run batch-size to increase batch size
CUDA_VISIBLE_DEVICES=4 python train.py data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --save-dir enc --adam-betas "(0.9, 0.98)" --max-epoch 100 --batch-size 256 --share-decoder-input-output-embed --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --eval-tokenized-bleu

for evaluation (run generate.py)
CUDA_VISIBLE_DEVICES=4 python generate.py data-bin/iwslt14.tokenized.de-en --path checkpoints/checkpoint100.pt --batch-size 128 --beam 5 --remove-bpe

example of results in different epochs:
data-bin/iwslt14.tokenized.de-en --path checkpoints/checkpoint100.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 34.45, 68.3/42.5/28.6/19.6 (BP=0.964, ratio=0.965, syslen=126568, reflen=131161)

data-bin/iwslt14.tokenized.de-en --path checkpoints/checkpoint20.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 33.73, 68.7/42.7/28.5/19.4 (BP=0.946, ratio=0.947, syslen=124242, reflen=131161)

data-bin/iwslt14.tokenized.de-en --path checkpoints/checkpoint10.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 31.52, 66.2/39.8/25.8/17.0 (BP=0.961, ratio=0.962, syslen=126167, reflen=131161)

data-bin/iwslt14.tokenized.de-en --path checkpoints/checkpoint5.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 25.23, 61.2/33.3/19.8/12.0 (BP=0.956, ratio=0.957, syslen=125522, reflen=131161)

for eval with masking heads (pass them as model-overrides:
added mask-layer-name "enc-enc or enc-dec or dec-dec"
mask-layer number of layer to mask the head of
mask-head number of head to mask

for example
CUDA_VISIBLE_DEVICES=4 python generate.py data-bin/iwslt14.tokenized.de-en --path checkpoints/checkpoint100.pt --batch-size 128 --beam 5 --remove-bpe --model-overrides "{'mask_layer': 5, 'mask_head': 3, 'mask_layer_name': 'enc-dec'}"

running with sandwich (with A,F configuration) for example:
CUDA_VISIBLE_DEVICES=4 python train.py data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --save-dir enc --adam-betas "(0.9, 0.98)" --max-epoch 100 --batch-size 256 --share-decoder-input-output-embed --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --eval-tokenized-bleu --enc-layer-configuration 'AAAFAFAFAFFF'
CUDA_VISIBLE_DEVICES=3 python train.py data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --save-dir dec --adam-betas "(0.9, 0.98)" --max-epoch 100 --batch-size 256 --share-decoder-input-output-embed --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --eval-tokenized-bleu --dec-layer-configuration 'AAAFAFAFAFFF'


