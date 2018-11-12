# **Winning a CFT 2018 spelling correction competition**
## **Or building a task-agnostic seq2seq pipeline**

![](https://pics.spark-in.me/upload/65e886af30f4e694d325bb431e1c41d3.jpg)
_This is the first time we managed to win (i.e. 1st place) an ML competition_

A small philosophic preamble.

We participated as Profi.ru team (your truly + Dmitry Voronin helped a bit) + Savva Kolbachev, my old time pal from many competitions, also a specialist in NLP.
It was a long, difficult, nervous and surprisingly rewarding journey for our team.

To sum all the experience in one line - **the journey is more important than the destination**. This competition taught us a lot about being persistent, trying new thigs, fallind down and standing up again.

I personally comlpiled / optimized 3 or 4 pipelines for this task, until I could build the best. Also, as usual, we found a crucial error one day before the finish line.

## **TLDR**

![](https://pics.spark-in.me/upload/bad1d7a0c9a0a3bc2289375f04f660bb.jpg)
We managed to apply some of the current state-of-the-art sequence-to-sequence models to a real `in the wild` [problem](https://datasouls.com/c/cft-contest/description#) in a challenging domain in a mixed multi-task / auto-encoder setting.

The core of the task - was input classification (**correct / needs to be corrected / random non-relevant trash**) and input correction.

The domain was - names of individuals from the countries of the CIS:
- Mostly Russia, Moldova, Uzbekistan, Georgia, Tadzikistan, Kazakhstan, Ukraine, etc;
- Names usually consist of 3 parts (name, surname, patronym). Sometimes in certain republics patronyms contain and additional suffix added as a fourth word;
- There were 2+ alphabets with a total around 70+ characters;

**Models that worked best**
- Tuned sequence-to-sequence bidirectional GRU (see details below) - `92.5%` wo heuristics, `93%` with heuristics;
- Proper sequence-to-sequence inference loop;
- Also a more plain model in keras where input is essentially mapped to output via TimeDistributedDense layer worked well;
- Strongest naive heuristic - if the output of seq2seq ingefence loop is the same as input - then the input is correct;

**Key takeaways:**
- Seq2seq transformer took **much more time to train** and we failed to make it works the same as more traditional models;
- Our main competitor (with ~200 submissions) did not use Deep Learning at all - he essentially reverse engineereed the augmentation pipeline of the hosts - which is tough work, but does not scale to other domains;
- Batch seq2seq predictions are key to fast inference;

## **Code release**

We will not bother replicating the whole training process

- Dockerfile in case you run into problems in your environment;
- When the competition started
- Keras model:
  - Link;
- PyTorch BiLSTM seq2seq:
  - Best model train utils;
  - Best model classes;
  - Best model train pipeline;
  - Typical launch command
- Text augmentations;


## **Domain description**

One image is better than a thousand words.
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
- Plain usage of library

![](https://pics.spark-in.me/upload/f8c096815176bff3b93a2c8741fec176.jpg)
The distribution of the number of errors



## **Ideas / solutions / models that mostly worked**
![](https://pics.spark-in.me/upload/bd857359ad78f8846dfad61fa8736e8a.jpg)

## **Ideas / solutions / models that did not work**


## **Text data augmentation**
Dima

![](https://pics.spark-in.me/upload/ed966651a0e7a34702fa0510089c40bc.jpg)

## **Key milestones in achieveing our result**

Original lm transformer
![](https://pics.spark-in.me/upload/32975595ab1ae7086183c9835536d6b8.jpg)

![](https://pics.spark-in.me/upload/647d3349c8f4e0d31572b1a31776ff7e.jpg)
![](https://pics.spark-in.me/upload/70fc52606e05cb457cb5d928ef4b85e6.jpg)
![](https://pics.spark-in.me/upload/5d37668131bf599590753ff8467a55b1.jpg)


![](https://pics.spark-in.me/upload/5d37668131bf599590753ff8467a55b1.jpg)

![](https://pics.spark-in.me/upload/fbeafcc4168866fd86809e71e316e319.jpg)


## **Post-processing heuristics**

### **Heuristic that worked**
Savva, AV
### **Heuristics that did not work, or that we did not explore**
Savva
![](https://pics.spark-in.me/upload/63bb287b0d3162de741cb07e687492d6.jpg)

## **Improving the annotated encoder-decoder and the annotated transformer**

As you may know

### **All of these abstractions from torchtext ... how to cook them well**
![](https://pics.spark-in.me/upload/a976bb75b686a17dc945f67be703fea1.jpg)

### **Batch predictions**

### **Loss weighting is crucial!**

### **KLD loss?**


## **Kind words about organization and comparison to Kaggle**


## **Competition pulse**
![](https://pics.spark-in.me/upload/3193fa61d2f05a14bbc048c504e6964b.jpg)
![](https://pics.spark-in.me/upload/75d5f0f8644fb932a423602955b3d8ed.jpg)
![](https://pics.spark-in.me/upload/aeb7ee6daecd667de8c6170bad6cfd0e.jpg)
![](https://pics.spark-in.me/upload/f90665508581c16155d3cffb65a6fb95.jpg)


## **References**

- Annotated encoder decoder and transformer
- OpenAI transformer
- Torch text data pipeline explained
