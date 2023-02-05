from collections import defaultdict
import os
import sys
import time
import copy
from tqdm.notebook import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from lossfunction import *
from models.unet_models import *
from torchsummary import summary
def train_model(model, dataloaders = None,optimizer=None, scheduler=None, num_epochs:int = 50, phase:str = None, task:str = None, lossfunction:str = None):
    if os.path.exists(os.path.join(os.getcwd(), 'task', task)):
        print("type in a new task name")
        task = input()
        os.makedirs(os.path.join(os.getcwd(), 'task', task))
    else:
        os.makedirs(os.path.join(os.getcwd(), 'task', task))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    torch.save(model, os.path.join(os.getcwd(), 'task', task, 'net.pkl'))

    optimizer = optimizer if optimizer else torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    try:
        if lossfunction=='FFL':
            criterion = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)

        elif lossfunction=='MSE':
            criterion = torch.nn.MSELoss()   

        elif lossfunction == 'BCE':
            criterion = torch.nn.CrossEntropyLoss()

        elif lossfunction == 'BCE-Dice':
            criterion = BceDiceLoss(multiclass=True, n_classes=model.n_classes)
            
        else:
            raise
    except:
        print('plz choose loss function')
        sys.exit(1)
        
    print(f'model: {summary(model, (3, 224, 224), dtypes=[torch.float])},\nepochs: {num_epochs},\nphase: {phase},\nloss: {criterion},\noptimizer: {optimizer}')
    print("waiting for input")
    input()
    print("continue")

    log_path = os.path.join(os.getcwd(), 'task', task, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    net_path = os.path.join(os.getcwd(), 'task', task, 'net')
    if not os.path.exists(net_path):
        os.makedirs(net_path)

    state_path = os.path.join(os.getcwd(), 'task', task, 'state')
    if not os.path.exists(state_path):
        os.makedirs(state_path)
    
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    avg_time = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for param_group in optimizer.param_groups:
            print('LR', param_group['lr'])
        since = time.time()
        ###################
        # training model #
        ###################
        model.train()
        train_loss = []
        mse = []
        bce = []
        epoch_samples = 0
        with torch.set_grad_enabled(True):
            for batch in tqdm(dataloaders['train']):

                image = batch['image'].to(device=device, dtype=torch.float)
                label = batch['mask'].to(device=device, dtype=torch.float)

                optimizer.zero_grad(set_to_none=True)
                
                # outputs, saliency_map = model(image)
                
                # mse_loss = F.mse_loss(saliency_map, torch.tensor(torch.mul(image, label), dtype=torch.float))
                # mse_loss = F.mse_loss(saliency_map, torch.tensor(label, dtype=torch.float))

                # FFL = FocalFrequencyLoss()
                # mse_loss = FFL(saliency_map, torch.tensor(torch.mul(image, label), dtype=torch.float))

                outputs = model(image)

                mse_loss = torch.tensor(0) #純unet

                '''
                bce+dice loss for segmentation unit
                '''
                bce_loss = criterion(outputs, torch.tensor(label, dtype=torch.long))
                
                loss = bce_loss + mse_loss
                # loss = ftl_loss + bce_loss
                train_loss.append(loss.data.cpu().numpy()*image.size(0))
                mse.append(mse_loss.data.cpu().numpy()*image.size(0))
                bce.append(bce_loss.data.cpu().numpy()*image.size(0))

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_samples += image.size(0)
            
            print('training: total loss: {:.6f}, mse loss: {:.6f}, bce loss: {:.6f}'.format(sum(train_loss) / epoch_samples, sum(mse) / epoch_samples, sum(bce) / epoch_samples))
            writer.add_scalar('Training/Loss', sum(train_loss) / epoch_samples, epoch)
            writer.add_scalar('Training/MSELoss', sum(mse) / epoch_samples, epoch)
            writer.add_scalar('Training/Dice+BCELoss', sum(bce) / epoch_samples, epoch)
        ###################
        # validation model #
        ###################
        model.eval()
        val_loss = []
        val_mse = []
        val_bce = []
        with torch.no_grad():
            epoch_samples = 0
            for batch in tqdm(dataloaders['val']):
                image = batch['image'].to(device=device, dtype=torch.float)
                label = batch['mask'].to(device=device, dtype=torch.float)

                # outputs, saliency_map = model(image)
                
                metrics = defaultdict(float)

                # mse_loss = F.mse_loss(saliency_map, torch.tensor(torch.mul(image, label), dtype=torch.float))
                # mse_loss = F.mse_loss(saliency_map, torch.tensor(label, dtype=torch.float))
                
                # FFL = FocalFrequencyLoss()

                # mse_loss = FFL(saliency_map, torch.tensor(torch.mul(image, label), dtype=torch.float))
                # FTL = FocalTversky_loss(tversky_kwargs={'alpha':0.5, 'beta':0.5})
                # ftl_loss = FTL(saliency_map, label)
                
                outputs = model(image)

                mse_loss = torch.tensor(0) #純unet

                '''
                bce+dice loss for segmentation unit
                '''
                bce_loss = criterion(outputs, torch.tensor(label, dtype=torch.long))

                loss = bce_loss + mse_loss

                val_loss.append(loss.data.cpu().numpy() * image.size(0))
                val_mse.append(mse_loss.data.cpu().numpy()*image.size(0))
                val_bce.append(bce_loss.data.cpu().numpy()*image.size(0))

                epoch_samples += image.size(0)
                
            print('validation: total loss: {:.6f}, mse loss: {:.6f}, bce loss: {:.6f}'.format(sum(val_loss) / epoch_samples, sum(val_mse) / epoch_samples, sum(val_bce) / epoch_samples))
            writer.add_scalar('Validation/Loss', sum(val_loss) / epoch_samples, epoch)
            writer.add_scalar('Validation/MSELoss', sum(val_mse) / epoch_samples, epoch)
            writer.add_scalar('Validation/Dice+BCELoss', sum(val_bce) / epoch_samples, epoch)
            epoch_loss = sum(val_bce) / epoch_samples

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, state_path + '/state-' + str(epoch+1) + '.pkl')

            if  epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        avg_time += time_elapsed
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}, take time: {}'.format(best_loss, avg_time))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), net_path + '/'+ 'best.pkl')
    
    with open(os.path.join(os.getcwd(), 'task', task, 'result.txt'), 'w') as f:
        f.write(f'model: {model},\nepochs: {num_epochs},\nphase: {phase},\nloss function by: {criterion},\nbest validation loss: {best_loss},\naverage time: {avg_time / num_epochs},\ntotal time: {avg_time}')

    return model