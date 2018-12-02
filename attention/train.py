import torch
from torch.autograd import Variable
import tqdm
import numpy as np

def train(attention_model,train_loader,dev_loader,criterion,optimizer,epochs = 5,use_regularization = False,C=0,clip=False):
    """
        Training code
 
        Args:
            attention_model : {object} model
            train_loader    : {DataLoader} training data loaded into a dataloader
            optimizer       :  optimizer
            criterion       :  loss function. Must be BCELoss for binary_classification and NLLLoss for multiclass
            epochs          : {int} number of epochs
            use_regularizer : {bool} use penalization or not
            C               : {int} penalization coeff
            clip            : {bool} use gradient clipping or not
       
        Returns:
            accuracy and losses of the model
 
      
        """
    attention_model = attention_model.cuda()
    criterion = criterion.cuda()
    losses = []
    accuracy = []
    for i in range(epochs):
        print("Running EPOCH",i+1)
        total_loss = 0.
        n_batches = 0.
        correct = 0.
        numIters = len(train_loader)
        qdar = tqdm.tqdm(enumerate(train_loader),
                                total=numIters,
                                ascii=True)
        for batch_idx,train in qdar: #enumerate(train_loader):

            attention_model.hidden_state = attention_model.init_hidden()
            x,y = Variable(train[0]).cuda(),Variable(train[1]).cuda()
            y_pred,att = attention_model(x)
           
            #penalization AAT - I
            if use_regularization:
                attT = att.transpose(1,2)
                identity = torch.eye(att.size(1))
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size,att.size(1),att.size(1))).cuda()
                penal = attention_model.l2_matrix_norm(att@attT - identity)
           
            
            if not bool(attention_model.type) :
                #binary classification
                #Adding a very small value to prevent BCELoss from outputting NaN's
                correct+=torch.eq(torch.round(y_pred.type(torch.cuda.DoubleTensor).squeeze(1)),y).data.sum()
                if use_regularization:
                    try:
                        #print(C * penal/train_loader.batch_size)
                        reg = C * penal/train_loader.batch_size                        
                        loss = criterion(y_pred.type(torch.cuda.DoubleTensor).squeeze(1),y) #+ C * penal/train_loader.batch_size
#                        print(reg)
 #                       print(reg.eq(torch.tensor(float('nan')).type(torch.cuda.DoubleTensor)))
  #                      if not reg.eq(torch.tensor(float('nan')).type(torch.cuda.DoubleTensor)):
                        loss += reg                       
                    except RuntimeError:
                        raise Exception("BCELoss gets nan values on regularization. Either remove regularization or add very small values")
                else:
                    loss = criterion(y_pred.type(torch.cuda.DoubleTensor).squeeze(1),y)
                
            
            else:
                
                correct+=torch.eq(torch.max(y_pred,1)[1],y.type(torch.cuda.LongTensor)).data.sum()
                if use_regularization:
                    loss = criterion(y_pred,y) + (C * penal/train_loader.batch_size).type(torch.cuda.FloatTensor)
                else:
                    loss = criterion(y_pred,y)
               
            qdar.set_postfix(loss=str(np.round(loss.data.item(),3)))

            total_loss+=loss.data
            optimizer.zero_grad()
            loss.backward()
           
            #gradient clipping
            if clip:
                torch.nn.utils.clip_grad_norm(attention_model.parameters(),0.5)
            optimizer.step()
            n_batches+=1
        cur_acc = correct.type(torch.cuda.FloatTensor)/(n_batches*train_loader.batch_size)
        print("avg_loss is",total_loss/n_batches)
        print("Accuracy of the model",cur_acc)
        losses.append(total_loss/n_batches)
        accuracy.append(cur_acc)

        correct_dev = 0
        numItersDev = len(dev_loader)
        qdarDev = tqdm.tqdm(enumerate(dev_loader),
                        total=numItersDev,
                        ascii=True)
        for batch_idx, dev in qdarDev:
            x_dev,y_dev = Variable(dev[0]).cuda(),Variable(dev[1]).cuda()
            y_pred_dev,att = attention_model(x_dev)
            if not bool(attention_model.type) :
                correct_dev+=torch.eq(torch.round(y_pred_dev.type(torch.cuda.DoubleTensor).squeeze(1)),y_dev).data.sum()
        cur_acc_dev = correct_dev.type(torch.cuda.FloatTensor)/(numItersDev*dev_loader.batch_size)
        print("Accuracy of the model on dev", cur_acc_dev)
    return losses,accuracy
 
 
def evaluate(attention_model,x_test,y_test):
    """
        cv results
 
        Args:
            attention_model : {object} model
            x_test          : {nplist} x_test
            y_test          : {nplist} y_test
       
        Returns:
            cv-accuracy
 
      
    """
   
    attention_model.batch_size = x_test.shape[0]
    attention_model.hidden_state = attention_model.init_hidden()
    x_test_var = Variable(torch.from_numpy(x_test).type(torch.LongTensor))
    y_test_pred,_ = attention_model(x_test_var)
    if bool(attention_model.type):
        y_preds = torch.max(y_test_pred,1)[1]
        y_test_var = Variable(torch.from_numpy(y_test).type(torch.LongTensor))
       
    else:
        y_preds = torch.round(y_test_pred.type(torch.DoubleTensor).squeeze(1))
        y_test_var = Variable(torch.from_numpy(y_test).type(torch.DoubleTensor))
       
    return torch.eq(y_preds,y_test_var).data.sum()/x_test_var.size(0)
 
def get_activation_wts(attention_model,x):
    """
        Get r attention heads
 
        Args:
            attention_model : {object} model
            x               : {torch.Variable} input whose weights we want
       
        Returns:
            r different attention weights
 
      
    """
    attention_model.batch_size = x.size(0)
    attention_model.hidden_state = attention_model.init_hidden()
    _,wts = attention_model(x)
    return wts
