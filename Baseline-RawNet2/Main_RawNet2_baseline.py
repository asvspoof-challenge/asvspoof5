import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from data_utils_RawNet2 import (Dataset_train,Dataset_dev,
                        Dataset_eval, genSpoof_list)
from model import RawNet2
from tensorboardX import SummaryWriter
from startup_config import set_random_seed
import torch.nn.functional as F
from utils import compute_eer
import logging
import time


def evaluate_accuracy(
    dev_loader,
    model,
    device):
    num_total = 0.0
    num_correct = 0.0
    model.eval()
    with torch.no_grad():
        label_loader, score_loader = [], []
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            _,batch_out = model(batch_x)
            _,batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            score = F.softmax(batch_out, dim=1)[:, 1]
            label_loader.append(batch_y)
            score_loader.append(score)
        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(label_loader, 0).data.cpu().numpy()
        val_eer = compute_eer(scores[labels == 1], scores[labels == 0])[0]  
        val_accuracy = (num_correct / num_total)*100
        return val_accuracy,val_eer*100


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
    model.eval()
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        _,batch_out = model(batch_x,is_test=True)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(
    train_loader,
    model,
    optimizer,
    device):

    """Training"""
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _,batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _,batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct/num_total)*100))
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof5 RawNet2-baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof5_database/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--output_dir', dest="output_dir",type=str,
                        default='./ASVspoof5_RawNet2_Baseline_model', help='Dir for Model checkpoint')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default='RawNet2_baseline_exp')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)

    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 

   
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    #define model saving path
    model_tag = 'model_{}_{}_{}_{}'.format(
        args.comment, args.num_epochs, args.batch_size, args.lr)
    model_save_path = os.path.join(args.output_dir, model_tag)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    #model 
    model = RawNet2(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    print('nb_params: {}'.format(nb_params))
    
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    #evaluation 
    if args.eval:
        file_eval = genSpoof_list( dir_meta = os.path.join(args.database_path+'ASVspoof5.eval.txt'),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set= Dataset_eval(list_IDs = file_eval, base_dir = args.database_path)
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

     
    # define train dataloader

    d_label_trn,file_train = genSpoof_list(dir_meta = os.path.join(args.database_path+'ASVspoof5.train.metadata.txt'),is_train=True,is_eval=False)
    print('no. of training trials',len(file_train))
    
    train_set = Dataset_train(list_IDs = file_train,
              labels = d_label_trn,
              base_dir = args.database_path)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=16, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta = os.path.join(args.database_path+'ASVspoof5.dev.metadata.txt'),is_train=False,is_eval=False)
    print('no. of validation trials',len(file_dev))

    dev_set = Dataset_dev(list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = args.database_path)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=16, shuffle=False)
    del dev_set,d_label_dev

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader,model,optimizer, device)
        val_accuracy, val_err = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('val_accuracy', val_accuracy, epoch)
        writer.add_scalar('training_loss', running_loss, epoch)
        writer.add_scalar('val_EER', val_err, epoch)
        print('\n{} - {} - {:.2f} - {:.2f} - {:.2f}'.format(epoch,
                                                   running_loss, train_accuracy, val_accuracy,val_err))
        logging.info(
            f"[{epoch}]: train_loss: {running_loss} - train_acc: {train_accuracy} - val_acc: {val_accuracy} - val_eer: {val_err}"

        )
        torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
        
        
