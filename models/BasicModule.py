import time

import torch as t


class BasicModule(t.nn.Module):
    """
    封装了nn.Module，主要提供save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        指定加载的模型
        :param path: 模型文件的路径
        :return: None
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名，
        如AlexNet_0710_23:57:29.pth
        :param name:
        :return: Default Format: ModelName_MONTHDAY_TIME.pth
        """
        if name is None:
            prefix = "checkpoints/" + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
