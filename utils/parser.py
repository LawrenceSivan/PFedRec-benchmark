import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('--clients_sample_num', type=int, default=0)
    parser.add_argument('--num_round', type=int, default=5)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--lr_eta', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='ml-100k')
    parser.add_argument('--num_users', type=int)
    parser.add_argument('--num_items', type=int)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--num_negative', type=int, default=4)
    parser.add_argument('--l2_regularization', type=float, default=0.)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--algorithm', type=str, default='pfedrec')
    parser.add_argument('--model_name', type=str, default='mlp', help='Internal model name, e.g. mlp')
    parser.add_argument('--similarity_metric', type=str, default='cosine')
    parser.add_argument('--neighborhood_size', type=int, default=10)
    parser.add_argument('--neighborhood_threshold', type=float, default=0.0)
    parser.add_argument('--mp_layers', type=int, default=1)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--layers', type=str, default='64, 32, 16, 8')
    args = parser.parse_args()

    if isinstance(args.layers, str):
        args.layers = [int(item) for item in args.layers.split(',')]

    return args

