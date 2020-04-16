this is the execution line for running the training
(set the eval-bleu-args for defult in translation.py because of argument issues
pass --eval-tokenized-bleu as well (otherwise bleu is not working)
pass max-epoch to decide the max epochs to run batch-size to increase batch size
CUDA_VISIBLE_DEVICES=4 python train.py data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --save-dir enc --adam-betas "(0.9, 0.98)" --max-epoch 100 --batch-size 256 --share-decoder-input-output-embed --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --eval-tokenized-bleu

for evaluation (run generate.py)
CUDA_VISIBLE_DEVICES=4 python generate.py data-bin/iwslt14.tokenized.de-en --path checkpoints/checkpoint50n.pt --batch-size 128 --beam 5 --remove-bpe

example of results in different epochs:
python generate.py data-bin/iwslt14.tokenized.de-en --path checkpoints_base_model/checkpoint100.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 34.45, 68.3/42.5/28.6/19.6 (BP=0.964, ratio=0.965, syslen=126568, reflen=131161)

python generate.py data-bin/iwslt14.tokenized.de-en --path checkpoints_base_model/checkpoint50.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 34.43, 68.8/43.0/28.9/19.8 (BP=0.954, ratio=0.955, syslen=125277, reflen=131161)

python generate.py data-bin/iwslt14.tokenized.de-en --path checkpoints_base_model/checkpoint20.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 33.73, 68.7/42.7/28.5/19.4 (BP=0.946, ratio=0.947, syslen=124242, reflen=131161)

python generate.py data-bin/iwslt14.tokenized.de-en --path checkpoints_base_model/checkpoint10.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 31.52, 66.2/39.8/25.8/17.0 (BP=0.961, ratio=0.962, syslen=126167, reflen=131161)

python generate.py data-bin/iwslt14.tokenized.de-en --path checkpoints_base_model/checkpoint5.pt --batch-size 128 --beam 5 --remove-bpe
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



use perplexity/ cross entropy loss
reshape line 347 to head and bz
max tokens 8096
test eval for masking with script for all options
set the seed and run the regular configuration 'AF' * 6 with regular launch
decoder all F before and all A last - see bad results (around 5bleu)

build instructions for ex:
go to this file and follow the train
go to transformer.py and check the args and follow
how to do the masking (start in transformer.py) and then to multi-headattention and apply multi head attention
applpy sandwich transformer --
(consider adding notebook)

goal: create draft (readme or notebook)



for config 'AAAFAFAFAFFF': only enc
data-bin/iwslt14.tokenized.de-en --path enc/checkpoint100.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 34.17, 67.7/41.9/28.0/19.2 (BP=0.972, ratio=0.973, syslen=127571, reflen=131161)

for config 'AAAFAFAFAFFF': only dec
data-bin/iwslt14.tokenized.de-en --path dec/checkpoint100.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 34.26, 68.1/42.3/28.2/19.2 (BP=0.969, ratio=0.969, syslen=127132, reflen=131161)


running with sandwich FFFFFFAAAAAA (encoder & decoder) for example:
CUDA_VISIBLE_DEVICES=6 python train.py data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --save-dir encoder_decoder_swap --adam-betas "(0.9, 0.98)" --max-epoch 50 --share-decoder-input-output-embed --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 12288 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --eval-tokenized-bleu --enc-layer-configuration 'FFFFFFAAAAAA' --dec-layer-configuration 'FFFFFFAAAAAA'

running normal configuration AFAFAFAFAFAF (encoder & decoder) for example:
CUDA_VISIBLE_DEVICES=1 python train.py data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --save-dir encoder_decoder_normal --adam-betas "(0.9, 0.98)" --max-epoch 50 --share-decoder-input-output-embed --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 12288 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --eval-tokenized-bleu --enc-layer-configuration 'AFAFAFAFAFAF' --dec-layer-configuration 'AFAFAFAFAFAF'

generate 
CUDA_VISIBLE_DEVICES=6 python generate.py data-bin/iwslt14.tokenized.de-en --path encoder_decoder_normal/checkpoint50.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 34.32, 68.8/42.9/28.8/19.7 (BP=0.954, ratio=0.955, syslen=125241, reflen=131161)

CUDA_VISIBLE_DEVICES=6 python generate.py data-bin/iwslt14.tokenized.de-en --path encoder_decoder_swap/checkpoint50.pt --batch-size 128 --beam 5 --remove-bpe
Generate test with beam=5: BLEU4 = 32.41, 68.4/41.8/27.5/18.4 (BP=0.934, ratio=0.936, syslen=122759, reflen=131161)

