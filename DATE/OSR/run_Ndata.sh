export CUDA_VISIBLE_DEVICES=2
for i in 65 60 55 50 45
do
python main.py --data real-n --semi_supervised 1 --sampling DATE --mode scratch --train_from 20140101 --test_from 20170101 --test_length 365 --valid_length 90 --initial_inspection_rate ${i} --final_inspection_rate 10 --epoch 20 --closs bce --rloss full --save 1 --numweeks 1 --inspection_plan direct_decay --batch_size 512 --device 2 --lr 0.005 --l2 0.001 --dim 32 --sample imp;
python main.py --data real-n --semi_supervised 1 --sampling DATE --mode scratch --train_from 20140101 --test_from 20170101 --test_length 365 --valid_length 90 --initial_inspection_rate ${i} --final_inspection_rate 10 --epoch 20 --closs bce --rloss full --save 1 --numweeks 1 --inspection_plan direct_decay --batch_size 512 --device 2 --lr 0.005 --l2 0.001 --dim 32 --sample hs;
done