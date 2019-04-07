# Structure_data_to_summary
Reproduce the result of  "Order-Planning Neural Text Generation From Structured Data" (arxiv link:https://arxiv.org/abs/1709.00155) paper for the task of generating natural Language summary from structured data.

For Inference:
python run_summarization.py --mode=decode --data_path=/home/anindya/Documents/Text_sum/finished_files/chunked/val_* --vocab_path=/home/anindya/Documents/Text_sum/finished_files/vocab --log_root=/home/anindya/Documents/Text_sum/ --exp_name=myexperiment

WILL CREATE SUMMARY IN /myexperiments/decode_val_400*/
where decode files give list of predicted summary.

where reference file gives all the list of corresponding reference summary or true summary..

python run_summarization.py --mode=train --data_path=/home/anindya/Documents/Text_sum/finished_files/chunked/train_* --vocab_path=/home/anindya/Documents/Text_sum/finished_files/vocab --log_root=/home/anindya/Documents/Text_sum/ --exp_name=myexperiment

For Training .. execute following command:
python run_order_planning_summarization.py --mode=train --data_path=/home/anindya/Documents/structure_data_to_summary/finished_files/chunked/train_* --vocab_path=/home/anindya/Documents/structure_data_to_summary/finished_files/ --log_root=/home/anindya/Documents/structure_data_to_summary/ --exp_name=myexperiment


For Inference.. execute following command:

python run_order_planning_summarization.py --mode=decode --data_path=/home/anindya/Documents/structure_data_to_summary/finished_files/chunked/val_* --vocab_path=/home/anindya/Documents/structure_data_to_summary/finished_files/ --log_root=/home/anindya/Documents/structure_data_to_summary/ --exp_name=myexperiment

Investigate weight of trained network:
python inspect_checkpoint.py /home/anindya/Documents/structure_data_to_summary/myexperiment/train/model.ckpt-2841

sample Results(generated summary from structure data) after epoch 1 of training in sample_output folder.

![alt text](https://github.com/anindyasarkarIITH/Structure_data_to_summary/master//home/anindya/Desktop/Structure_data_to_Summary/sample_output/sample_output4.png)
