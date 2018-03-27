import argparse

def parse_args(type=0):
    if type == 0:
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
        parser.add_argument('--data', default='data/sick/',
                            help='path to dataset')
        parser.add_argument('--glove', default='data/glove/',
                            help='directory with GLOVE embeddings')
        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=15, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=0.01, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--optim', default='adam',
                            help='optimizer (default: adam)')
        parser.add_argument('--seed', default=123, type=int,
                            help='random seed (default: 123)')
        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
        cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        parser.set_defaults(cuda=True)

        args = parser.parse_args()
        return args
    else: # for sentiment classification on SST
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analysis Trees')
        parser.add_argument('--name', default='default_name',
                            help='name for log and saved models')
        parser.add_argument('--saved', default='saved_model',
                            help='name for log and saved models')

        parser.add_argument('--model_name', default='constituency',
                            help='model name constituency or dependency')
        parser.add_argument('--data', default='data/sst/',
                            help='path to dataset')
        parser.add_argument('--glove', default='data/glove/',
                            help='directory with GLOVE embeddings')
        parser.add_argument('--batchsize', default=1, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=30, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=0.05, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--emblr', default=0, type=float,
                            metavar='EMLR', help='initial embedding learning rate')
        parser.add_argument('--wd', default=0, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--reg', default=0, type=float,
                            help='l2 regularization (default: 1e-4)')
        parser.add_argument('--optim', default='adagrad',
                            help='optimizer (default: adagrad)')
        parser.add_argument('--seed', default=123, type=int,
                            help='random seed (default: 123)')
        parser.add_argument('--fine_grain', default=1, type=int,
                            help='fine grained (default 0 - binary mode)')
                            # untest on fine_grain yet.
        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
        cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        cuda_parser.add_argument('--lower', dest='cuda', action='store_true')
        parser.set_defaults(cuda=True)
        parser.set_defaults(lower=True)

        args = parser.parse_args()
        return args
