import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Configuration for the model')

    # Add arguments based on the config dictionary
    parser.add_argument('--root_data', type=str, default="/home/nas2_userH/hyesulim/Data",
                        help='Root path for the data')
    parser.add_argument('--root_code', type=str, default="/home/nas2_userH/hyesulim/Dev/2023/11785-f23-prj/linear_probe",
                        help='Root path for the code')
    # For device, you might want to handle it differently
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--obj', type=str, default='layer1_2_relu3',
                        help='Objective')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--subset_samples', type=int, default=10000,
                        help='Number of subset samples')
    parser.add_argument('--train', type=bool, default=True,
                        help='Training mode')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Test model checkpoint directory')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
