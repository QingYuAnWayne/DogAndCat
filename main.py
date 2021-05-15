from data.dataset import DogCat
import models
from config import DefaultConfig
import torch as t
from utils.visualizer import Visualizer
from torch.utils.data import DataLoader
from torchnet import meter
from torch.autograd import Variable

opt = DefaultConfig()


def train(**kwargs):
    """
    训练
    训练的主要步骤如下：
    - 定义网络
    - 定义数据
    - 定义损失函数和优化器
    - 计算重要指标
    - 开始训练
      - 训练网络
      - 可视化各种指标
      - 计算在验证集上的指标
    :param kwargs:
    :return:
    """
    # 根据命令行更新参数
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # Step 1 定义网络
    model = getattr(models, opt.model)()
    # model = models.ResNet34()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2: 数据
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step3: 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 0.0

    # 训练
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        ii = 0
        for data, label in train_dataloader:
            if opt.use_gpu:
                data = data.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            score = model(data)
            loss = criterion(score, label.long())
            loss.backward()
            optimizer.step()

            # 更新统计指标和可视化
            num = loss.data.item()
            loss_meter.add(num)
            confusion_matrix.add(score.data, label.data)
            if ii % opt.print_feq == opt.print_feq - 1:
                vis.plot("loss", loss_meter.value()[0])
            ii += 1

        model.save()

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}"
            .format(
            epoch=epoch,
            loss=loss_meter.value()[0],
            val_cm=str(val_cm.value()),
            train_cm=str(confusion_matrix.value()),
            lr=lr))

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    """
    检验
    :param model:
    :param dataloader:
    :return:
    """
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
    with t.no_grad():
        for ii, (data, label) in enumerate(dataloader):
            input = t.FLoatTensor(data)
            label = t.FloatTensor(label)
            if opt.use_gpu:
                input = input.cuda()
                label = label.cuda()
            score = model(input)
            confusion_matrix.add(score.data.squeeze(), label.long())

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. *(cm_value[0][0] + cm_value[1][1]) / cm_value.sum()
    return confusion_matrix, accuracy


def test(**kwargs):
    """
    测试
    :param kwargs:
    :return:
    """
    pass


def help():
    """
    打印帮助信息
    :return:
    """
    print("""
       usage : python {0} <function> [--args=value,]
       <function> := train | test | help
       example: 
               python {0} train --env='env0701' --lr=0.01
               python {0} test --dataset='path/to/dataset/root/'
               python {0} help
       avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    # import fire
    #
    # fire.Fire()
    train()
