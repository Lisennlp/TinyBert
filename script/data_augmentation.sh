

python data_augmentation.py  \
        --pretrained_bert_model  /nas/pretrain-bert/pretrain-pytorch/bert-base-uncased  \
        --data_path  data/en_data.txt    \
        --glove_embs  /nas/lishengping/datas/glove.6B.300d.txt   \
        --M  15    \
        --N  30    \
        --p  0.4   3>&2 2>&1 1>&3 | tee logs/data_augmentation.log