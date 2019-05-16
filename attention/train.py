import torch
from torch.autograd import Variable
import tqdm
import numpy as np

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

def train(attention_model,train_loader,criterion,optimizer1,optimizer2,x_test, y_test, epochs = 5,use_regularization = False,C=0,clip=False):
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
    if torch.cuda.is_available():
        attention_model = attention_model.cuda()
        criterion = criterion.cuda()
    losses = []
    accuracy = []

    def lstr(loss):
        return str(np.round(loss.data.item(),3))

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

            x,y = Variable(train[0]),Variable(train[1])
            if torch.cuda.is_available():
                x,y = x.cuda(), y.cuda()
            

            # branch 1
            y_pred, y_pred_msk, y_pred_adv = attention_model(x,0,y)
            correct+=torch.eq(torch.max(y_pred,1)[1],y.type(device.LongTensor)).data.sum()
            loss1_pred = criterion(y_pred, y.type(device.LongTensor))
            loss1_msk = criterion(y_pred_msk, y.type(device.LongTensor)) 
            loss1_adv = criterion(y_pred_adv, 1-y.type(device.LongTensor)) 
            loss1 = loss1_pred + loss1_msk + loss1_adv
            
            total_loss+=loss1.data
            optimizer1.zero_grad()
            loss1.backward()
            #gradient clipping
            if clip:
                torch.nn.utils.clip_grad_norm(attention_model.parameters(),0.5)
            optimizer1.step()

            # branch 2
            y_pred, y_pred_msk, y_pred_adv = attention_model(x,1,y)
            correct+=torch.eq(torch.max(y_pred,1)[1],y.type(device.LongTensor)).data.sum()
            loss2_pred = criterion(y_pred, y.type(device.LongTensor))
            loss2_msk = criterion(y_pred_msk, y.type(device.LongTensor)) 
            loss2_adv = criterion(y_pred_adv, 1-y.type(device.LongTensor)) 
            loss2 = loss2_pred + loss2_msk + loss2_adv
            
            total_loss+=loss2.data
            optimizer2.zero_grad()
            loss2.backward()
            #gradient clipping
            if clip:
                torch.nn.utils.clip_grad_norm(attention_model.parameters(),0.5)
            optimizer2.step()

            qdar.set_postfix(ls1p=lstr(loss1_pred), ls1msk=lstr(loss1_msk), ls1adv=lstr(loss1_adv), 
                                ls2p=lstr(loss2_pred), ls2msk=lstr(loss2_msk), ls1adv=lstr(loss2_adv))
            n_batches+=1
        cur_acc = correct.type(device.FloatTensor)/(2*n_batches*train_loader.batch_size)
        print("avg_loss is",total_loss/(2*n_batches))
        print("Accuracy of the model",cur_acc)
        losses.append(total_loss/(2*n_batches))
        accuracy.append(cur_acc)
        # evaluate
        with torch.no_grad():
            eval_acc = evaluate(attention_model,x_test,y_test)
        print("test accuracy is "+str(eval_acc))
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
   
    bsize_bak = attention_model.batch_size
    attention_model.batch_size = x_test.shape[0]
    attention_model.hidden_state = attention_model.init_hidden()
    x_test_var = Variable(torch.from_numpy(x_test).type(device.LongTensor))
    y_test_pred = attention_model(x_test_var)
    # if bool(attention_model.type):
    y_preds = torch.max(y_test_pred,1)[1]
    y_test_var = Variable(torch.from_numpy(y_test).type(device.LongTensor))
    
    attention_model.batch_size = bsize_bak
    attention_model.hidden_state = attention_model.init_hidden()
    # else:
    # y_preds = torch.round(y_test_pred.type(device.DoubleTensor).squeeze(1))
    # y_test_var = Variable(torch.from_numpy(y_test).type(device.DoubleTensor))
       
    return torch.eq(y_preds,y_test_var).data.sum().type(device.DoubleTensor)/x_test_var.size(0)
 
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
    pred = attention_model(x)
    att = attention_model.getAttention(torch.max(pred,1)[1])
    return att
