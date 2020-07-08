# TinyBERT

本项目是基于华为的TinyBert进行修改的，简化了数据读取的过程，方便我们利用自己的数据进行读取操作。  


TinyBert的训练过程：  
- 1、用通用的Bert base进行蒸馏，得到一个通用的student model base版本；  
- 2、用相关任务的数据对Bert进行fine-tune得到fine-tune的Bert base模型；  
- 3、用2得到的模型再继续蒸馏得到fine-tune的student model base，注意这一步的student model base要用1中通用的student model base去初始化；（词向量loss + 隐层loss + attention loss）  
- 4、重复第3步，但student model base模型初始化用的是3得到的student模型。（任务的预测label loss）


General Distillation （通用版预训练语言模型蒸馏）
====================
- 预训练

```
sh script/general_train.sh
                             
```
Task Distillation （fine-tune版预训练语言模型蒸馏）
====================
- 预训练

```
# 第一阶段
sh script/task_train.sh one

# 第二阶段
sh script/task_train.sh two
                             
```

## 数据格式  

    data/*.txt

## 数据增强

    sh script/augmentation.sh

- 参数说明：

    python data_augmentation.py  \   
            --pretrained_bert_model  /nas/pretrain-bert/pretrain-pytorch/bert-base-uncased  \   # bert预训练模型   
            --data_path  data/en_data.txt    \  # 需要增强的数据路径   
            --glove_embs  /nas/lishengping/datas/glove.6B.300d.txt   \   # glove词向量文件   
            --M  15    \  # 从文本中选择M个词可能被替换   
            --N  30    \  # 通过bert mask预测的概率前N个词去替换   
            --p  0.4   \  # 某个词被替换的概率   
            3>&2 2>&1 1>&3 | tee logs/data_augmentation.log

论文在fine-tune阶段采用了数据增强的策略，从后面的实验中可以看出，数据增强起到了很重要的作用。   
**数据扩充的过程如下: **
对于特定任务的数据中每一条文本，首先使用bert自带的方式进行bpe分词，bpe分词之后是完整单词（single-piece word），用[MASK]符号代替，然后使用bert进行预测并选择其对应的候选词N个；如果bpe（就是BertTokenizer）分词之后不是完整单词，则使用Glove词向量以及余弦相似度来选择对应的N个候选词，最后以概率p选择是否替换这个单词，从而产生更多的文本数据。  

- 数据

    因为懒得找中文的词向量下载，暂时用英文的glove文件和英文原始数据。中文的话只需要把预训练模型改为中文bert，glove改为中文的词向量文件即可。原始数据样式和增强后的数据样式在data/en_data.txt和data/aug_en_data.txt


## Evaluation  

待续...


## 官方版本

=================1st version to reproduce our results in the paper ===========================

[General_TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj) 

[General_TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1wXWR00EHK-Eb7pbyw0VP234i2JTnjJ-x)

=================2nd version (2019/11/18) trained with more (book+wiki) and no `[MASK]` corpus =======

[General_TinyBERT_v2(4layer-312dim)](https://drive.google.com/open?id=1PhI73thKoLU2iliasJmlQXBav3v33-8z)

[General_TinyBERT_v2(6layer-768dim)](https://drive.google.com/open?id=1r2bmEsQe4jUBrzJknnNaBJQDgiRKmQjF)


We here also provide the distilled TinyBERT(both 4layer-312dim and 6layer-768dim) of all GLUE tasks for evaluation. Every task has its own folder where the corresponding model has been saved.

[TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1_sCARNCgOZZFiWTSgNbE7viW_G5vIXYg) 

[TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1Vf0ZnMhtZFUE0XoD3hTXc6QtHwKr_PwS)

