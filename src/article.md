# **Winning a CFT 2018 spelling correction competition**
## **Or building a task-agnostic seq2seq pipeline on a challenging domain**

![](https://pics.spark-in.me/upload/65e886af30f4e694d325bb431e1c41d3.jpg)
_This is the first time we managed to win (i.e. 1st place) an ML competition_

A small philosophic preamble.

We participated as [Profi.ru](https://profi.ru/) team (yours truly + Dmitry Voronin helped a bit) and Sava Kalbachou ([Lucidworks](https://lucidworks.com/)), my old time pal from many competitions, also a specialist in NLP.
It was a long, difficult, nervous and surprisingly rewarding journey for our team.

To sum all the experience in one line - **the journey is more important than the destination**. This competition taught us a lot about being persistent, trying new things, falling down and standing up again.

I personally compiled / optimized 3 or 4 pipelines for this task, until I could build the best one. Also, as usual, we found a crucial error one day before the finish line xD

## **TLDR**

![](https://pics.spark-in.me/upload/bad1d7a0c9a0a3bc2289375f04f660bb.jpg)
We managed to apply some of the current state-of-the-art sequence-to-sequence models to a real `in the wild` [problem](https://datasouls.com/c/cft-contest/description#) in a challenging domain in a mixed multi-task / auto-encoder setting.

The core of the task was - input classification (**correct / needs to be corrected / random non-relevant trash**) and input correction.

The domain was - names of individuals from the countries of the CIS:
- Mostly Russia, Moldova, Uzbekistan, Georgia, Tadzhikistan, Kazakhstan, Ukraine, etc;
- Names usually consist of 3 parts (name, surname, patronym) - `Иванов Иван Иванович`. Sometimes in certain republics patronyms have additional suffix added as a fourth word (`ОГЛЫ`, same as `вич` suffix in Russian, but a separate word);
- There were 2+ alphabets with a total around 70+ characters + a lot of noise in the data;

**Models that worked best**
- Sequence-to-sequence:
  - Tuned sequence-to-sequence bidirectional GRU (see details below) - `92.5%` wo heuristics, `93%` with heuristics, `93%+` in ensemble;
  - Proper sequence-to-sequence inference loop;
- Also a simpler model in Keras without bottleneck and with Conv1D output layer worked well - with a top score of `88-89%`;
- Strongest naïve heuristic - if the output of seq2seq inference loop is the same as input - then the input is correct;

**Key takeaways:**
- Seq2seq transformer took **much more time to train** and we failed to make it work the same as traditional models. Obviously we lack experience with it, but this is the fourth time I notice **real  people** mention that transformer is not for "real" people (i.e. non-practical);
- Our main competitor (with ~200 submissions) did not use Deep Learning at all - he **essentially reverse engineered the augmentation pipeline of the hosts** - which is tough work, but does not scale to other domains;
- Batch seq2seq predictions are key to fast inference;
- Beam search, though being a tough gem to implement, **boosted the score insignificantly** (obvious culprit - small dictionary);

## **Code release**

We will not bother replicating the whole training process, since there were a lot of fine-tuning steps, but overall **these bits will be reusable for you**:

- [Preparing](https://github.com/snakers4/cft-contest-2018/blob/master/src/preprocessing_av.ipynb) the competition dataset (see the first chapter of the notebook);
- [Dockerfile](https://github.com/snakers4/cft-contest-2018/blob/master/Dockerfile) in case you run into problems in your environment (when the image was build PyTorch 4.0 or 4.1 was a major version, target it, or you will have to make code compatible);
- Keras model:
  - Some [preprocessing](https://github.com/snakers4/cft-contest-2018/blob/master/savva_final_solution/preprocessing.py);
  - Best model train [utils](https://github.com/snakers4/cft-contest-2018/blob/master/savva_final_solution/local_utils.py);
  - Best model train [pipeline](https://github.com/snakers4/cft-contest-2018/blob/master/savva_final_solution/spell_simple_seq2seq_train_predict.py);
  - Typical launch commands:
```
	python preprocessing.py
	python spell_simple_seq2seq_train_predict.py
```
- PyTorch BiLSTM seq2seq:
  - Best model train [utils](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/encoder_decoder_utils.py);
  - Best model [classes](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/encoder_decoder.py);
  - Best model train [pipeline](https://github.com/snakers4/cft-contest-2018/blob/master/src/train_encoder_decoder_predict.py);
  - Typical launch command (flag names are self-explanatory):
```
CUDA_VISIBLE_DEVICES=0 python3 train_encoder_decoder_predict.py \
	--resume weights/pred_eval_encdec_savva_l4_heavy_h512e512_1e4_aug2fxpseudo_lrf01_resume3_best.pth.tar \
	--heavy_decoder True --lr_factor 0.5 \
	--trn_df_path ../data/pseudo_aug2_proc_trn.csv \
	--tb_name encdec_savva_l4_heavy_h512e512_1e4_aug2fxpseudo_lrf01_resume4 \
	--lr 1e-6 --epochs 100 \
	--batch_size 128 \
	--hidden_size 512 --dropout 0.2 \
	--num_layers 4 --num_classes 3  --emb_size 512 \
	--cn_emb_size 0 --num_cn 0 \
	--tensorboard True \
```
- Text [augmentations](https://github.com/snakers4/cft-contest-2018/blob/master/src/generate_mistakes.py);
- Preparing the best augmented [datasets](https://github.com/snakers4/cft-contest-2018/blob/master/src/final_aug.py);
- Naïve post-processing for single model;


## **Domain description**

One illustration is better than a thousand words.
Please also note - in train + test there were ~2000 countries, i.e. country input was also very noisy.

id | fullname | country | target | fullname_true
-- | -- | -- | -- | --
1003765 | ГОИБОВ ЭЛЬДАР | РОССИЯ | 0 | NaN
1814726 | ХОПЕРИА СЕРГО | ГРУЗИЯ | 0 | NaN
1937537 | ИСМАИЛОВ АРИФ ИСМАИЛ ОГЛЫ | РОССИЯ | 0 | NaN
490273 | ХАЛИФАЕВ РАХМАТУЛЛО МАРДОНХОДЖАЕВИЧ | РОССИЯ | 0 | NaN
763314 | НУРОВ МЕХРОНШОХ ПИРБОЕВИЧ | РОССИЯ | 0 | NaN
1227264 | СТЕБАЕВА ЕЛЕНА АЛЕКСАНДРОВНА | УКРАИНА | 0 | NaN
1212322 | НАСРУЗОВ ШЕРАЛИ ХАМИТОВИЧ | УЗБЕКИСТАН | 1 | НАВРУЗОВ ШЕРАЛИ ХАМИТОВИЧ
170238 | ОТАЖАГОВАСАНОБАР ИЛХОМБОЕВНА | УЗБЕКИСТАН | 1 | ОТАЖАНОВА САНОБАР ИЛХОМБОЕВНА
614604 | АЛИКУЛОВ ДАВРОН ЖУМАЕВИЧ | РОССИЯ | 0 | NaN
1650211 | КАЮМОВ АЛЕКСЕЙ МУРОДОВИПЧ | RUS | 1 | КАЮМОВ АЛЕКСЕЙ МУРОДОВИЧ
1180543 | ПОЫУЕВ АЛЕКСАНДР АНАТОЛЬЕВИЧ | КЫРГЫЗСТАН | 1 | ПОЧУЕВ АЛЕКСАНДР АНАТОЛЬЕВИЧ
18050 | КАЗАКОВА ВЕРА ВАЛЕОЬЕВНА | UKR | 1 | КАЗАКОВА ВЕРА ВАЛЕРЬЕВНА
183279 | АНОШКА ИРИАНВЛАДИМИРОВНА | РОССИЯ | 1 | АНОШКА ИРИНА ВЛАДИМИРОВНА
740820 | ДЖАББОРОВА ОЙМИНОС КОСИМАЛИЕВНА | ТАДЖИКИСТАН | 0 | NaN
1722259 | САЙДАЛИЕВ БАХРИДИН ЗАЙНУДИНОВИЧ | РОССИЯ | 0 | NaN
1527516 | ЖРАЕВ АХРОРИДИН МАДАМИНЖОН УГЛИ | РОССИЯ | 1 | ЖУРАЕВ АХРОРИДИН МАДАМИНЖОН УГЛИ
1208578 | ХАЛИМОВ МИХАИЛ ШЕРАЛИЕВИЧ | РОССИЯ | 0 | NaN
252703 | САЙКО ИРИНА | MOLDOVA | 0 | NaN
1365562 | DRAGANOVVALERIU VIUCTOR | MOLDOVA | 1 | DRAGANOV VALERIU VICTOR
1963028 | ЗОКОРОВ НЕКРУЗ | РОССИЯ | 0 | NaN

The hosts of the competition also shared a strong and simple [baseline](https://github.com/datasouls/cft2018-fiotypos/blob/master/baseline/baseline.ipynb):
- A linear regression classifier on ~`1m` n-grams;
- Plain usage of 3rd party library to correct misprints;

![](https://pics.spark-in.me/upload/f8c096815176bff3b93a2c8741fec176.jpg)
_The distribution of the number of errors_



## **Ideas / solutions / models that worked**
![](https://pics.spark-in.me/upload/bd857359ad78f8846dfad61fa8736e8a.jpg)


Please refer to these **amazing articles** for more nuts and bolts of the models we used:
- Mechanics of seq2seq models with [attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/);
- The amazing illustrated [transformer](http://jalammar.github.io/illustrated-transformer/);
- Annotated [encoder decoder](https://bastings.github.io/annotated_encoder_decoder/) and [annotated transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (older PyTorch);

**What worked:**
- Sequence to sequence modelling with biLSTMs (91%);
- Validation matched LB precisely - we did not try K-Fold validation, but splitting the dataset in advance was accurate enough;
- Using a simple pipeline to clean extra characters and obvious errors;
- Pseudo-labelling and text augmentations (91% + 1.5% on a single model);
- Using the "whole" train dataset, i.e. using all of the correct names as "correct" and using all of the incorrect names. Paired with additional augmentations (1x size of the train dataset) - turned out to be our single best pre-processing pipeline (91% + 1.5% on a single model);
- Post-processing heuristics (see below);
- Running batch-based inference loops (in the annotated encoder decoder and transformer predictions are handled with batch-size of 1 - sic!);

## **Ideas / solutions / models that did not work or we did not try**

- [BPE / Sentencepiece](https://github.com/google/sentencepiece) - we tried using several vocabularies (500, 1500, 5000 tokens size etc.) obtained in unsupervised fasion via Sentencepiece. But it didn't really help model to converge faster and results were worse than we expected.
- [Beam search](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/encoder_decoder_utils.py#L353) - most likely because of model confidence and small vocabulary. It boosted the score a bit, but did not provide a lot of value. It was a tough nut to crack, but I was dissapointed it did not work as much as we expected;
- Transformer model - when I adopted and optimized the annotated transformer [pipeline](http://nlp.seas.harvard.edu/2018/04/03/attention.html) it worked, but failed to converge within meaningful time;
- LM assisted classifier from Open AI transformer experiments - it failed to converge at all, most likely because it lacked a proper seq2seq inference loop, it was a low hanging fruit for me (I just had the this pipeline ready, while Sava boasted his initial 88% pipeline);
- Multi-GPU / cluster / distributed training - didn't try it;
- Running augmentations on the fly - I believe you can extend this [line](https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L257) in torchtext to do so, but I postponed it a bit, and then just had no time to do that. But that may speed up the convergence 3-10x from my experience;

## **Text data augmentation**
After some reverse engineering process (by the way based on the word ngrams), it turned out, that organizers use pretty simple typos generation methods. There are 5 main types:
- __insert__, the most dominant type (_~50%_ of all cases). Just takes a random letter in a word and inserts a new letter before or after it. A new letter is a random closest (on a keybord) letter to the chosen one, for example - letters __W__, __S__ or __A__ if __Q__ is chosen. __IVAN__ - __ICVAN__
- __change__, _~30%_, just changes a random letter in a word into random closest to it on a keybord (like in __insert__ case). __IVAN__ - __IVQN__
- __swap__ positions of two nearest random letters, _~5%_. __IVAN__ - __IAVN__
- __delete space__ between two words, _~7.5%_
- __delete letter__ (completely random _~7.5%_), __IVAN__ - __VAN__

Each query with typos contains a combination of one (_93%_ of all cases), two (_6%_) or three (_~1%_) mistakes listed above.

So we managed to write our own funcion for generating an additional dataset and use it alongside the original one for training.


![](https://pics.spark-in.me/upload/ed966651a0e7a34702fa0510089c40bc.jpg)

## **Key milestones in achieving our result**


Model | Overall score | Plain heuristic boost | Comment
-- | -- | -- | --
OpenAI transformer | <50% | NA | we abandoned this idea (`*`)
Baseline | <80% | NA |
Baseline naïve Keras seq2seq | 88-89% | NA | it took Sava ~3 hours to build this
BiGRU baseline seq2seq | 82-85% | NA | following annotated encoder decoder
Best transformer seq2seq | 85%+ | NA | too long to train `(**)`
BiGRU seq2seq + tuning | 89-90% | 1% | `(***)`
BiGRU seq2seq + tuning + countries | 89-90% | 1% | no improvement
BiGRU seq2seq + tuning + heavy decoder | ~91% | 1% | minor improvement
BiGRU seq2seq + tuning + heavy decoder +   skip | ~91% | 1% | minor or no improvement
BiGRU seq2seq + tuning + heavy decoder +   WRONG train / psedo augs | ~91.5% | 0.5%+ |
BiGRU seq2seq + tuning + heavy decoder +   CORRECT train / psedo augs | >92.5% | 0.5%+ | `(****)`

- `(*)` Most likely we just did not invest enough time into this, but the target of the base pipeline base different, so we abandoned the idea;
- `(**)` Whereas the biggest reasonable BiGRU model trained for 15-20 hours, this would train for at least a week on one GPU to achieve similar result;
- `(***)` LR decay mostly, MOAR layers!;
- `(****)` Two time we made errors when generating augmented data. At first I mixed 2 columns (source and target) in 50% of cases - and the model trained a bit better at first. Then I forgot to change target labels when augmenting the data


## **End to end metric curves**

**Original lm transformer**
![](https://pics.spark-in.me/upload/32975595ab1ae7086183c9835536d6b8.jpg)

**Original encoder decoder seq2seq**
![](https://pics.spark-in.me/upload/647d3349c8f4e0d31572b1a31776ff7e.jpg)

**Tuned encoder decoder seq2seq**
![](https://pics.spark-in.me/upload/70fc52606e05cb457cb5d928ef4b85e6.jpg)

**Adding WRONG augmentations**
![](https://pics.spark-in.me/upload/5d37668131bf599590753ff8467a55b1.jpg)

**Playing with lower lr**
![](https://pics.spark-in.me/upload/74d6fbc2815244e78c06f37f14682ed2.jpg)

**Adding CORRECT augmentations**
![](https://pics.spark-in.me/upload/fbeafcc4168866fd86809e71e316e319.jpg)


## **Post-processing heuristics**

### **Heuristic that worked**

After building the first seq2seq model we noticed that sometimes model predicts class 0 (phrase doesn't contains error), but predicted phrase is different from original one. We used this as a heuristic to change predicted class from 0 to 1 (phrase contains error), that gave us 0.5-1.5% boost, depending on the quality of the model (less for strong models).

Another heuristic was in changing predicted class/phrase to predictions from quite different model (like Keras one) in cases when we are sure that there is a mistake in predictions (model predicts class 1, but predicted phrase is equal to original phrase).

### **Heuristics that did not work, or that we did not explore**

At some moment of time we noticed, that the network is several times more likely to make errors with the words it "knows" compared to "new" words (names / surnames / patronyms). It meant that the network generalized well, but nevertheless we tried applying an obvious heuristic to test, in essence similar to pseudo-labelling.

Obviously we tried using dictionary heuristic, i.e. checking whether the predicted words are present in the vocabulary of names. It gave a significant boost on validation, but failed to provide reasonable boost on test (most likely because there are rare names that are predicted wrong and / or are not present in train). Obviously we tried adding predicted "correct" test names, but that did not help.

![](https://pics.spark-in.me/upload/63bb287b0d3162de741cb07e687492d6.jpg)
Train vs. test word dictionary overlap

## **Ensembling models**

As it often happens during competitions, at some point closely to the end you just have to do some stacking/ensembling to continue compete and stay at the top of LB. This contest wasn't an exception.

We used hold-out validation data to find optimal way of ensembling and its parameters. For the first task of classification we used kind of geometrical mean of predicted probabilities from variety diverse models. For the second taks of name correction we implemented simple majority voting mechanism that was applied for each of the token of predicted names. After each step heuristics described above were used as well.

This quite simple yet efficient ensemble gave us final ~0.003% boost and 1st place. We selected it and best single model as our two final submissions:
![](https://pics.spark-in.me/upload/c83b820499d1080452a2160fb7c9d685.png)

## **Improving the annotated encoder-decoder and the annotated transformer**

As you may know, biLSTMs / biGRUs and transformers are the best models on the market when it comes to the challenging tasks like the following:
- Neural Machine Translation;
- Text abstraction;
- Sequence to sequence modelling;
- Text inference tasks;

And we were standing on the shoulders of the giants when re-using the stellar pipelines from these tutorials [encoder decoder](https://bastings.github.io/annotated_encoder_decoder/) and [annotated transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).

But as everything in life, they are not perfect.

**Key shortcomings:**
- Predictions in both of them are handled with batch-size of one. This is not ideal when you run a lot of end-to-end experiments;
- Annotated encoder decoder boasts loss weighting, but it is not explained that when pairing this with secondary objectives (like in our case) - you have to be extra careful with weighting both components of the loss to avoid over / underfitting on one task;
- While torchtext is awesome for the purpose it was build (to accelerate NMT experiments) - for applied competition NLP pipelines - it is a bit slow and bulky and seed parameters are non intuitive. Probably I should have solved caching a bit better, but I opted for the simplest solution with CSV files I explain below;
- I fully understand that distributed / data parallel routines in PyTorch is a bit experimental and in high maintenance, but ofc annotated transformers multi GPU loss compute function broke in `PyToch 0.4+`. I decided not to fix this, but probably I should have, expecially for transformer;

### **Batch predictions and beam search**

Much attention in both turotials was paid to running models efficiently when training and validating the model. But for some reason the inference greedy search loops presented in the tutorials are written for batches with size 1.

As you may be aware, this slows down end-to-end validation 10x. So, I have written essentially 3 functions:
- Batch based prediction loops for [encoder decoder](https://github.com/snakers4/cft-contest-2018/blob/master/src/train_ann_transformer.py#L466) and [transformer](https://github.com/snakers4/cft-contest-2018/blob/master/src/train_encoder_decoder_predict.py#L503);
- Batch-based greedy search for [encoder decoder](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/encoder_decoder_utils.py#L131) and [transformer](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/ann_transformer_utils.py#L96);
- [Beam search](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/encoder_decoder_utils.py#L353) for encoder decoder;

### **All of these abstractions from torchtext ... how to cook them well**
![](https://pics.spark-in.me/upload/a976bb75b686a17dc945f67be703fea1.jpg)


### **Model flavours**

At some point in time I noticed that there are essentially 2 bottle-necks in our best model:
- The flow between encoder and decoder is bottle-necked by the fact that encoder is bidirectional, and decoder is not. So simply speaking I solved this by making my decoder [2 times bigger](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/encoder_decoder.py#L22) (and it helped my model to converge);
- I also experimented with adding [country embeddings](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/encoder_decoder.py#L64) and adding a [skip connection](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/encoder_decoder.py#L36) and one more attention layer in essense similar to copy attention from abstraction models - these ideas did not work for me;
- I also played with essentially [dropping](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/encoder_decoder.py#L53) embeddings at all (char level model), with poor results; On the other hand, Keras models worked reasonably better at char level than using embeddings.

Notably, best models were quite heavy, heavy decoder also adding quite a lot of time to epoch.

### **Loss weighting is crucial!**

I did not run ablation tests for this (I had a lot of experience failing in object detection with 2-3 losses, though) - but I assumed that since loss is essentially applied `n_token` times for language modelling target and then normalized by the number or tokens, to achieve best results the classification loss should be similary [sized](https://github.com/snakers4/cft-contest-2018/blob/master/src/pytorch/encoder_decoder_utils.py#L89). I achieved this simply by using the most convenient loss layer in PyTorch (yeah, I forgot to divide this loss by batch-size, but this turned out to be not so cricial, I guess).

### **KLD loss?**

A neat idea from annotated transformer that I did not explore - replace the classification loss with KLD-like loss (usually used in variational auto-encoders) to add more variance to predictions. But I guess it is not important for a small vocabulary.

## **Kind words about organization and comparison to Kaggle**

If you do Kaggle / monitor the ML scene, then you may be aware that Kaggle is all of these things:
- Essentially now it is a joke for serious people;
- Plagued by bozo explosion;
- Each recent major competition features a severe idiotic leak and/or organization flaw - obviously no one cares there anymore;
- In 2018 it is essentially great to get **free experience / free datasets / free code** - but all the medals and achievements are largely overrated (kernels are mostly farmed by guys who wants to get a lot of attention, but sometimes there are super stellar pipelines);

**Do not believe me - just read these posts** [1](https://www.kaggle.com/c/airbus-ship-detection/discussion/64355) [2](https://www.kaggle.com/c/airbus-ship-detection/discussion/64366) [3](https://www.kaggle.com/c/airbus-ship-detection/discussion/64393);

Anyway to the point - the organization of this competition was NOTHING like Kaggle:
- No leaks and / or opportunities for idiotic LB probing;
- Good correlation of local validation / private / public LB;
- Only a few minor erros by hosts (occasional glitches with the platform, no team formation deadline, too late deadline for exernal data, no compute limitations and no dockerization / re-running requirements);
- Interesting and challenging domain;

## **Competition pulse**

At first Sava dominated the leaderboard (**LB**) with his 88% score for a long time.
This is the moment when competition caught up. My pipelines mostly were failures at that moment.

![](https://pics.spark-in.me/upload/3193fa61d2f05a14bbc048c504e6964b.jpg)

Then I finally managed to make the annotated encoder decoder work and we were trolling the LB again for some time (probably a week or so).

![](https://pics.spark-in.me/upload/75d5f0f8644fb932a423602955b3d8ed.jpg)

We did not really bother to stack our model and / or show our best results, but at this point it was worrying, because our single best model + heuristics did not really have a lot of breathing room. At this point we could achieve ~`92%`, but we were waiting and trying different models and heuristics.

![](https://pics.spark-in.me/upload/aeb7ee6daecd667de8c6170bad6cfd0e.jpg)

This was pre-final results sometime during the last day. Notice `thousandvoices` was at the top. We had to ensemble our models quite a bit here. Also I found a bug in my augmentation pipeline 1 day before the competition end, and our single best model started learning quite quickly and achieved ~`92.5%` w/o heuristics and ~`93%` with a plain heuristic.

But this did not transfer well to our ensemble. 1% gain in the best model translated into `0.2-0.3%` for reasons explained above.

![](https://pics.spark-in.me/upload/d629b2eae0b0886b3217f02b13d68ebd.jpg)

It was quite a hot battle during the last minutes (I wonder why people do not sleep at 11:59 PM on Sunday???)

![](https://pics.spark-in.me/upload/92f69dbdbc696dbc9d74b719dd7a4f42.jpg)

And this is the final private leaderboard. Notice that we kept our ground and `thousandvoices` likely:
- Either somehow over-fitted (or maybe he used some leaks, who knows);
- Or just forgot to choose his best final submissions, who knows;

But history is written by the victors.

![](https://pics.spark-in.me/upload/f90665508581c16155d3cffb65a6fb95.jpg)


## **References**
- Mechanics of seq2seq models with [attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/);
- The amazing illustrated [transformer](http://jalammar.github.io/illustrated-transformer/);
- Annotated [encoder decoder](https://bastings.github.io/annotated_encoder_decoder/) and [annotated transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (older PyTorch);
- OpenAI transformer [implementation](https://github.com/huggingface/pytorch-openai-transformer-lm) in PyTorch;
- Torch text data pipeline [explained](http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/);
- A [case](https://medium.com/analytics-vidhya/https-medium-com-tomkenter-why-care-about-byte-level-seq2seq-models-in-nlp-26bcf05dd7d3) for byte level NMT models;
