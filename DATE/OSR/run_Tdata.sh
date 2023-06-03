export CUDA_VISIBLE_DEVICES=0
for i in 65 60 55 50 45
do
python main.py --data real-t --semi_supervised 1 --sampling DATE --mode scratch --train_from 20170101 --test_from 20190101 --test_length 365 --valid_length 90 --initial_inspection_rate ${i} --final_inspection_rate 10 --epoch 20 --closs bce --rloss full --save 1 --numweeks 1 --inspection_plan direct_decay --batch_size 1024 --device 0 --lr 0.005 --l2 0 --dim 32 --sample imp;
python main.py --data real-t --semi_supervised 1 --sampling DATE --mode scratch --train_from 20170101 --test_from 20190101 --test_length 365 --valid_length 90 --initial_inspection_rate ${i} --final_inspection_rate 10 --epoch 20 --closs bce --rloss full --save 1 --numweeks 1 --inspection_plan direct_decay --batch_size 1024 --device 0 --lr 0.005 --l2 0 --dim 32 --sample hs;
done