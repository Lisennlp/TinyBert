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
sh script/task_train.sh one
                             
```

## 数据格式  

    data/*.txt


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


## Evaluation  

待续...