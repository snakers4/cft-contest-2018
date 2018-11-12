# **Winning a CFT 2018 spelling correction competition**
## **Or building a task-agnostic seq2seq pipeline**

![](https://pics.spark-in.me/upload/65e886af30f4e694d325bb431e1c41d3.jpg)
_This is the first time we managed to win (i.e. 1st place) an ML competition_

A small philosophic preamble.

We participated as Profi.ru team (your truly + Dmitry Voronin helped a bit) + Savva Kolbachev, my old time pal from many competitions, also a specialist in NLP.
It was a long, difficult, nervous and surprisingly rewarding journey for our team.

To sum all the experience in one line - **the journey is more important than the destination**. This competition taught us a lot about being persistent, trying new thigs, fallind down and standing up again.

I personally compiled / optimized 3 or 4 pipelines for this task, until I could build the best one. Also, as usual, we found a crucial error one day before the finish line xD

## **TLDR**

![](https://pics.spark-in.me/upload/bad1d7a0c9a0a3bc2289375f04f660bb.jpg)
We managed to apply some of the current state-of-the-art sequence-to-sequence models to a real `in the wild` [problem](https://datasouls.com/c/cft-contest/description#) in a challenging domain in a mixed multi-task / auto-encoder setting.

The core of the task - was input classification (**correct / needs to be corrected / random non-relevant trash**) and input correction.

The domain was - names of individuals from the countries of the CIS:
- Mostly Russia, Moldova, Uzbekistan, Georgia, Tadzikistan, Kazakhstan, Ukraine, etc;
- Names usually consist of 3 parts (name, surname, patronym) - `Иванов Иван Иванович`. Sometimes in certain republics patronyms contain and additional suffix added as a fourth word (`ОГЛЫ`, same as `вич` suffix in Russian, but a separate word);
- There were 2+ alphabets with a total around 70+ characters + a lot of noise in the data;

**Models that worked best**
- Sequence-to-sequence:
  - Tuned sequence-to-sequence bidirectional GRU (see details below) - `92.5%` wo heuristics, `93%` with heuristics, `93%+` in ensemble;
  - Proper sequence-to-sequence inference loop;
- Also a more plain model in keras where input is essentially mapped to output via TimeDistributedDense layer worked well;
- Strongest naive heuristic - if the output of seq2seq ingefence loop is the same as input - then the input is correct;

**Key takeaways:**
- Seq2seq transformer took **much more time to train** and we failed to make it works the same as more traditional models. Obciously we lacked experience with it, but this is the fourth time I notice real  people mention that transformer is not for "real" people (i.e. for Google with 512 TPUs);
- Our main competitor (with ~200 submissions) did not use Deep Learning at all - he **essentially reverse engineereed the augmentation pipeline of the hosts** - which is tough work, but does not scale to other domains;
- Batch seq2seq predictions are key to fast inference;
- Beam search, though being a tough gem to implement, boosted the score insignificantly (obvious culprit - small dictionary);

## **Code release**

We will not bother replicating the whole training process, since there were a lot of fine-tuning steps, but overall these bits will be reusable for you:

- Dockerfile in case you run into problems in your environment (when the image was build PyTorch 4.0 or 4.1 was a major version, target it, or you will have to make code compatible);
- Keras model:
  - Link;
- PyTorch BiLSTM seq2seq:
  - Best model train utils;
  - Best model classes;
  - Best model train pipeline;
  - Typical launch command
- Text augmentations;
- Post-processing;
- Naive post-processing for single model;


## **Domain description**

One illustration is better than a thousand words.
Please also note - in train + test there were ~2000 counties, i.e. country input was also very noisy.

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

The hosts of the competition also shared a strong [baseline](https://github.com/datasouls/cft2018-fiotypos/blob/master/baseline/baseline.ipynb):
- A linear regression classifier on 1m n-grams;
- Plain usage of 3rd party library to correct misprints;

![](https://pics.spark-in.me/upload/f8c096815176bff3b93a2c8741fec176.jpg)
_The distribution of the number of errors_



## **Ideas / solutions / models that mostly worked**
![](https://pics.spark-in.me/upload/bd857359ad78f8846dfad61fa8736e8a.jpg)

## **Ideas / solutions / models that did not work**


## **Text data augmentation**
Dima

![](https://pics.spark-in.me/upload/ed966651a0e7a34702fa0510089c40bc.jpg)

## **Key milestones in achieveing our result**


Model | Overall score | Plain heuristic boost | Comment
-- | -- | -- | --
OpenAI transformer | <50% | NA | we abandoned this idea (`*`)
Baseline | <80% | NA |
Baseline naïve keras seq2seq | 88% | NA | it took Savva 3 hours to build this
BiGRU baseline seq2seq | 82-85% | NA | following annotated encoder decoder
Best transformer seq2seq | 85%+ | NA | too long to train `(**)`
BiGRU seq2seq + tuning | 89-90% | 1% | `(***)`
BiGRU seq2seq + tuning + countries | 89-90% | 1% | no improvement
BiGRU seq2seq + tuning + heavy decoder | ~91% | 1% | minor improvement
BiGRU seq2seq + tuning + heavy decoder +   skip | ~91% | 1% | minor or no improvement
BiGRU seq2seq + tuning + heavy decoder +   WRONG train / psedo augs | ~91.5% | 0.5%+ |
BiGRU seq2seq + tuning + heavy decoder +   CORRECT train / psedo augs | >92.5% | 0.5%+ | `(****)`

- `(*)` Most likely we just did not invest enough time into this, but the target of the base pipeline base different, so we abandoned the idea
- `(**)` Whereas the fattest reasonable BiGRU model trained for 15-20 hours, this would train for at least a week on one GPU to achieve similar result;
- `(***)` LR decay mostly, MOAR layers!;
- `(****)` Two time we made erros when generating augmented data. At first I mixed 2 columns (source and target) in 50% of cases - and the model trained a bit better at first. Then I forgot to change target labels when augmenting the data


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
Savva, AV

After building the first seq2seq model we noticed that sometimes the model predicted class 1 (with errors), but failed to change the phrase. We used this as a heuristic, that gave us from 0.5% boost to 1% boost, depending on the quality of the model (worse for strong models).

### **Heuristics that did not work, or that we did not explore**

At some moment of time we noticed, that the network is several times more likely to make errors with the words it "knows" compared to "new" words (names / surnames / patronyms). It meant that the network generalized well, but nevertheless we tried applying an obvious heuristic to test, in essence similar to pseudo-labelling.

Obviously we tried using dictionary heuristic, i.e. checking whether the predicted words are present in the vocabulary of names. It gave a significant boost on validation, but failed to provide reasonable boost on test (most likely because there are rare names that are predicted wrong and / or are not present in train). Obviosuly we tried adding predicted "correct" test names, but that did not help.

![](https://pics.spark-in.me/upload/63bb287b0d3162de741cb07e687492d6.jpg)
Train vs. test word dictionary overlap

Savva

## **Improving the annotated encoder-decoder and the annotated transformer**

As you may know

### **All of these abstractions from torchtext ... how to cook them well**
![](https://pics.spark-in.me/upload/a976bb75b686a17dc945f67be703fea1.jpg)

### **Batch predictions**

### **Loss weighting is crucial!**

### **KLD loss?**

## **Kind words about organization and comparison to Kaggle**

## **Competition pulse**

At first Savva dominated the leaderboard (**LB**) with his 88% score for a long time.
This is the moment when competition caugh up. My pipelines mostly were failures at this moment.

![](https://pics.spark-in.me/upload/3193fa61d2f05a14bbc048c504e6964b.jpg)

Then I finally manage to make the annotated encoder decoder work and we were trolling the LB again for some time (probably a week or so).

![](https://pics.spark-in.me/upload/75d5f0f8644fb932a423602955b3d8ed.jpg)

We did not really bother to stack our model and / or show our best results, but at this point it was worrying, because our single best model + heuristics did not really have a lot of breathing room. At this point we could achieve ~`92%`, but we waited and tried different models and heuristics.

![](https://pics.spark-in.me/upload/aeb7ee6daecd667de8c6170bad6cfd0e.jpg)

This was pre-final result sometime during the last day. Notice thousandvoices is at the top. We had to ensemble our models quite a bit here. Also I found a bug in my augmentation pipeline 1 day before the competition end, and our single best model started learning quite quickly and achieved ~`92.5%` w/o heuristics and ~`93%` with a plain heuristic.

But this did not transfer well to our ensemble. 1% gain in the best model translated into `0.2-0.3%` for reasons explained above.

![](https://pics.spark-in.me/upload/d629b2eae0b0886b3217f02b13d68ebd.jpg)

It was quite a heated battle during the last minutes (I wonder why people do not sleep at 11:59 PM on Sunday???)

![](https://pics.spark-in.me/upload/92f69dbdbc696dbc9d74b719dd7a4f42.jpg)

And this is the final private leaderboard. Notice that we kept our ground and thousandvoices likely:
- Either somehow overfitted (or maybe he used some leaks, who knows);
- Or just forgot to choose his final submission, who knows;

But it is victors, who write history.

![](https://pics.spark-in.me/upload/f90665508581c16155d3cffb65a6fb95.jpg)


## **References**

- Annotated [encoder decoder](https://bastings.github.io/annotated_encoder_decoder/) and [annotated transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (older PyTorch);
- OpenAI transformer
- Torch text data pipeline [explained](http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/);
