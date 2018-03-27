from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
from utils import map_label_to_target, map_label_to_target_sentiment
import torch.nn.functional as F

class SentimentTrainer(object):
    """
    For Sentiment module
    """
    def __init__(self, args, model, embedding_model ,criterion, optimizer):
        super(SentimentTrainer, self).__init__()
        self.args       = args
        self.model      = model
        self.embedding_model = embedding_model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
#        self.embedding_model.train()
#        self.embedding_model.zero_grad()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        # torch.manual_seed(789)
#        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
#            tree, sent, label = dataset[indices[idx]]
            tree, sent, label = dataset[idx]
            input = Var(sent)
            target = Var(map_label_to_target_sentiment(label,dataset.num_classes, fine_grain=self.args.fine_grain))
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input), 1)
            output, err = self.model.forward(tree, emb, training = True)
            #params = self.model.childsumtreelstm.getParameters()
            # params_norm = params.norm()
            err = err/self.args.batchsize # + 0.5*self.args.reg*params_norm*params_norm # custom bias
            loss += err.data[0] #
            err.backward()
            k += 1
            if k==self.args.batchsize:
#                for f in self.embedding_model.parameters():
#                    f.data.sub_(f.grad.data * self.args.emblr)
                self.optimizer.step()
#                self.embedding_model.zero_grad()
                self.optimizer.zero_grad()
                k = 0
        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.embedding_model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        predictions = predictions
        indices = torch.range(1,dataset.num_classes)
        for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
            tree, sent, label = dataset[idx]
            input = Var(sent, volatile=True)
            target = Var(map_label_to_target_sentiment(label,dataset.num_classes, fine_grain=self.args.fine_grain), volatile=True)
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input),1)
            output, _ = self.model(tree, emb) # size(1,5)
            err = self.criterion(output, target)
            loss += err.data[0]
            output[:,1] = -9999 # no need middle (neutral) value
            val, pred = torch.max(output, 1)
            predictions[idx] = pred.data.cpu()[0][0]
            # predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), predictions


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
            ltree,lsent,rtree,rsent,label = dataset[indices[idx]]
            linput, rinput = Var(lsent), Var(rsent)
            target = Var(map_label_to_target(label,dataset.num_classes))
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree,linput,rtree,rinput)
            err = self.criterion(output, target)
            loss += err.data[0]
            err.backward()
            k += 1
            if k%self.args.batchsize==0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.range(1,dataset.num_classes)
        for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
            ltree,lsent,rtree,rsent,label = dataset[idx]
            linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            target = Var(map_label_to_target(label,dataset.num_classes), volatile=True)
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree,linput,rtree,rinput)
            err = self.criterion(output, target)
            loss += err.data[0]
            predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), predictions
