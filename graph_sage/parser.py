import argparse
import time

curr_time = str(round(time.time(),3))

def get_parser():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters related to DATE
    parser.add_argument('--epoch', type=int, default=20, help="Number of epochs for DATE-related models")
    parser.add_argument('--batch_size', type=int, default=10000, help="Batch size for DATE-related models")
    parser.add_argument('--dim', type=int, default=16, help="Hidden layer dimension")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--l2', type=float, default=0.01, help="l2 reg")
    parser.add_argument('--alpha', type=float, default=10, help="Regression loss weight")
    parser.add_argument('--head_num', type=int, default=4, help="Number of heads for self attention")
    parser.add_argument('--use_self', type=int, default=1, help="Whether to use self attention")
    parser.add_argument('--fusion', type=str, choices=["concat","attention"], default="concat", help="Fusion method for final embedding")
    parser.add_argument('--agg', type=str, choices=["sum","max","mean"], default="sum", help="Aggreate type for leaf embedding")
    parser.add_argument('--act', type=str, choices=["mish","relu"], default="relu", help="Activation function")
    
    # Hyperparameters related to customs selection
    parser.add_argument('--devices', type=str, default=['0','1','2','3'], help="list of gpu available")
    parser.add_argument('--device', type=int, default=1, help='select which device to run, choose gpu number in your devices or cpu') 
    parser.add_argument('--output', type=str, default="result"+"-"+curr_time, help="Name of output file")
    parser.add_argument('--sampling', type=str, default = 'bATE', choices=['random', 'xgb', 'xgb_lr', 'DATE', 'diversity', 'badge', 'bATE', 'upDATE', 'enhanced_bATE', 'hybrid', 'tabnet', 'ssl_ae', 'noupDATE', 'randomupDATE'], help='Sampling strategy')
    parser.add_argument('--initial_inspection_rate', type=float, default=10, help='Initial inspection rate in training data by percentile')
    parser.add_argument('--final_inspection_rate', type=float, default = 5, help='Percentage of test data need to query')
    parser.add_argument('--inspection_plan', type=str, default = 'direct_decay', choices=['direct_decay','linear_decay','fast_linear_decay'], help='Inspection rate decaying option for simulation time')
    parser.add_argument('--mode', type=str, default = 'finetune', choices = ['finetune', 'scratch'], help = 'finetune last model or train from scratch')
    parser.add_argument('--subsamplings', type=str, default = 'bATE/DATE', help = 'available for hybrid sampling, the list of sub-sampling techniques seperated by /')
    parser.add_argument('--weights', type=str, default = '0.5/0.5', help = 'available for hybrid sampling, the list of weights for sub-sampling techniques seperated by /')
    parser.add_argument('--uncertainty', type=str, default = 'naive', choices = ['naive', 'self-supervised'], help = 'Uncertainty principle : ambiguity of illicitness or self-supervised manner prediction')
    parser.add_argument('--rev_func', type=str, default = 'log', choices = ['log'], help = 'Uncertainty principle : ambiguity of illicitness or self-supervised manner prediction')
    parser.add_argument('--closs', type=str, default = 'bce', choices = ['bce', 'focal'], help = 'Classification loss function')
    parser.add_argument('--rloss', type=str, default = 'full', choices = ['full', 'masked'], help = 'Regression loss function')
    parser.add_argument('--train_from', type=str, default = '20130101', help = 'Training period start from (YYYYMMDD)')
    parser.add_argument('--test_from', type=str, default = '20130201', help = 'Testing period start from (YYYYMMDD)')
    parser.add_argument('--test_length', type=int, default=7, help='Single testing period length (e.g., 7)')
    parser.add_argument('--valid_length', type=int, default=7, help='Validation period length (e.g., 7)')
    parser.add_argument('--data', type=str, default='synthetic', choices = ['synthetic', 'real-n', 'real-m', 'real-t', 'real-k', 'real-c'], help = 'Dataset')
    parser.add_argument('--numweeks', type=int, default=50, help='number of test weeks (week if test_length = 7)')
    parser.add_argument('--semi_supervised', type=int, default=0, help='Additionally using uninspected, unlabeled data (1=semi-supervised, 0=fully-supervised)')
    parser.add_argument('--identifier', type=str, default=curr_time, help='identifier for each execution')
    parser.add_argument('--save', type=int, default=0, help='Save intermediary files (1=save, 0=not save)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for spliting dataset')
    
    return parser