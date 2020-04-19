
# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

To install fairseq:
```bash
pip install fairseq
```

## Training a baseline model on IWSLT'14 German to English
In this part of the excersice, we will train a baseline model which we will use in the next parts

The following instructions can be used to train a Transformer model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

First download and preprocess the data:
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

Next we'll train a Transformer translation model over this data:
```
CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --save-dir baseline
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 12288 \
    --best-checkpoint-metric ppl 
    --maximize-best-checkpoint-metric \
    --fp16
```

--max-tokens can be reduced in order to use less gpu memory
we use perplexity (--best-checkpoint-metric ppl) as checkpoint metric
--save-dir baseline (save into baseline folder)
--fp16 train with mixed precision (if the machine support mixed presition training, not a must)

Finally we can evaluate our trained model:
```
CUDA_VISIBLE_DEVICES=0 python generate.py data-bin/iwslt14.tokenized.de-en \
    --path baseline/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```

the perplexity for the best model (after 50 epochs) should be ****

## Masking heads for the baseline model

In this part of the excersice, we will see the effect of masking different heads in the transformer layers.

```
CUDA_VISIBLE_DEVICES=0 python generate.py data-bin/iwslt14.tokenized.de-en \
    --path baseline/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
    --model-overrides "{'mask_layer': 5, 'mask_head': 3, 'mask_layer_name': 'enc-dec'}"
```
mask_layer is the layer number to mask
mask_head is the head number to mask
mask_layer_name is the name of the attention to mask - 'enc-enc' is the transformer encoder self attention
													 - 'enc-dec' is the transformer decoder cross attention
													 - 'dec-dec' is the transformer decoder self attention

follow this arguments to see their impact.

in the end, the mask_head argument, turn into head_to_mask on the function forward in the multihead_attention.py file.
your task is to implement the mask part inside the forward function (1-3 lines)

**option 1: create a script to launch all the evalutation with all the possibilities of masking (or maybe they will use my script)**

validate your results and explain them.

## Training a sandwitch model on IWSLT'14 German to English

In this part of the excersice, we will see the effect of different configuration of the transformers.
As mentioned in lecture 3, both the encoder transformer and the decoder transformer contain Multi head attention part - MHA (in the decoder we reffer to
both self attention and cross attention) followed by 2 layer feed forward netowork - FFN

we reffer the regular configuration as AFAFAFAFAFAF (we mark MHA layer as A and the FFN part layer as F)
and now we will check another configuration, and see it's result

```
CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --save-dir baseline
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 12288 \
    --best-checkpoint-metric ppl 
    --maximize-best-checkpoint-metric \
    --fp16 \
    --enc-layer-configuration 'FFFFFFAAAAAA'
    --dec-layer-configuration 'FFFFFFAAAAAA'
```

follow the enc-layer-configuration and dec-layer-configuration arguments and implement TransformerEncoderLayerFFN TransformerEncoderLayerMHA 
TransformerDecoderLayerMHA and TransformerEncoderLayerFFN
