
# 第一阶段
if [ "$1" = "one" ];then
        echo 'Run one stage train...'
        CUDA_VISIBLE_DEVICES=2,3 python task_distill.py   \
                          --teacher_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/ \
                          --student_model /home/lishengping/caiyun_projects/TinyBERT/student_model/output_dir  \
                          --data_dir  /nas/lishengping/datas/tiny_task_data \
                          --do_lower_case \
                          --do_train  \
                          --train_batch_size 20 \
                          --num_labels 2  \
                          --output_dir ./output_dir  \
                          --learning_rate 5e-5  \
                          --num_train_epochs  3  \
                          --eval_step  5000  \
                          --max_seq_len  128  \
                          --gradient_accumulation_steps  1  3>&2 2>&1 1>&3 | tee logs/tiny_bert.log


# 第二阶段
elif [ "$1" = "two" ];then
        echo 'Run two stage train...'

        CUDA_VISIBLE_DEVICES=2,3 python task_distill.py   \
                          --teacher_model /nas/pretrain-bert/pretrain-pytorch/chinese_wwm_ext_pytorch/ \
                          --student_model /home/lishengping/caiyun_projects/TinyBERT/student_model/output_dir  \
                          --data_dir  /nas/lishengping/datas/tiny_task_data \
                          --do_lower_case \
                          --do_train  \
                          --train_batch_size 20 \
                          --num_labels 2  \
                          --pred_distill  \
                          --task_mode  classification  \
                          --output_dir ./output_dir  \
                          --learning_rate 3e-5  \
                          --num_train_epochs  3  \
                          --eval_step  5000  \
                          --max_seq_len  128  \
                          --gradient_accumulation_steps  1  3>&2 2>&1 1>&3 | tee logs/tiny_bert.log

else
    echo 'unknown argment 1'
fi