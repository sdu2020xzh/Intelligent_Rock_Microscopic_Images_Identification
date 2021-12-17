import warnings
from torch import nn,optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.dataloader import *
from models.model import *
from utils import *
from visualizations.vis import Visualizer

#1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
# def net(num_classes):


#2. evaluate func
def evaluate(val_loader,model,criterion,epoch):
    #2.1 define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    #progress bar
    val_progressor = ProgressBar(mode="Val  ",
                                 epoch=epoch,
                                 total_epoch=config.epochs,
                                 model_name=config.model_name,total=len(val_loader))
    #2.2 switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i,(input,target) in enumerate(val_loader):
            val_progressor.current = i
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            #target = Variable(target).cuda()
            #2.2.1 compute output
            output = model(input)
            output = model.fc(output)
            loss = criterion(output,target)

            #2.2.2 measure accuracy and record loss
            precision1,precision2 = accuracy(output,target,topk=(1,2))
            losses.update(loss.item(),input.size(0))
            top1.update(precision1[0],input.size(0))


            # vis.plot('val_loss', losses.avg)
            # vis.plot('val_precision', top1.avg)

            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()
        val_progressor.done()
    return [losses.avg,top1.avg]


def main():
    fold = 0
    #4.1 mkdirs
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep +str(fold) + os.sep)



    # vis = Visualizer(env=config.model_name)
    # 创建模型
    model = torchvision.models.densenet121(pretrained=True)
    # 全连接层
    model.fc = nn.Linear(1000, config.num_classes)
    model.cuda()


    optimizer = optim.Adam(model.parameters(),
                           lr = config.lr,
                           amsgrad=True,
                           weight_decay=config.weight_decay)
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0
    best_precision1 = 0
    best_precision_save = 0

    # 读取数据
    train_data_list = get_files(config.train_data,"train")
    val_data_list = get_files(config.val_data,"val")

    train_dataloader = DataLoader(ChaojieDataset(train_data_list),
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  pin_memory=True,
                                  num_workers=8)
    val_dataloader = DataLoader(ChaojieDataset(val_data_list,train=False),
                                batch_size=config.batch_size*2,
                                shuffle=True,
                                collate_fn=collate_fn,
                                pin_memory=False,
                                num_workers=8)

    scheduler =  optim.lr_scheduler.StepLR(optimizer,
                                           step_size = 10,
                                           gamma=0.1)
    #4.5.5.1 define metrics
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    valid_loss = [np.inf,0,0]
    model.train()

    #4.5.5 train
    for epoch in range(start_epoch,config.epochs):
        scheduler.step(epoch)
        # 定义进度条
        train_progressor = ProgressBar(mode="Train",epoch=epoch,
                                       total_epoch=config.epochs,
                                       model_name=config.model_name,
                                       total=len(train_dataloader))

        # 训练
        for iter,(input,target) in enumerate(train_dataloader):
            train_progressor.current = iter
            model.train()

            # 定义输入图像
            input = Variable(input).cuda()
            # 定义标注信息
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            # 神经网络输出
            output = model(input)
            output = model.fc(output)
            # 计算损失
            loss = criterion(output,target)

            precision1_train,precision2_train = accuracy(output,target,
                                                         topk=(1,2))
            train_losses.update(loss.item(),input.size(0))
            train_top1.update(precision1_train[0],input.size(0))

            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg

            #if (iter + 1) % config.plot_every == 0:

            # vis.plot('train_loss', train_losses.avg)
            # vis.plot('train_precision', train_top1.avg)

            # 梯度反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 显示进度条
            train_progressor()

        train_progressor.done()


        #evaluate
        #lr = get_learning_rate(optimizer)

        #evaluate every half epoch
        valid_loss = evaluate(val_dataloader,model,criterion,epoch)
        is_best = valid_loss[1] > best_precision1
        best_precision1 = max(valid_loss[1],best_precision1)

        try:
            best_precision_save = best_precision1.cpu().data.numpy()
        except:
            pass

        save_checkpoint({
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_precision1":best_precision1,
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "valid_loss":valid_loss,
        },is_best,fold)

if __name__ =="__main__":
    main()





















