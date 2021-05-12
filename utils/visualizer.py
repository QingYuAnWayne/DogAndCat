import visdom
import time
import numpy as np

class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    比如
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    """
    def __init__(self, env="default", **kwargs):
        """
        画的第几个数，相当于横坐标
        保存（’loss',23） 即loss的第23个点
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        :param env:
        :param kwargs:
        :return:
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss', 1.00)
        :param name:
        :param y: 坐标
        :param kwargs:
        :return:
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def plot_many(self, d):
        """
        一次性plot多个！
        :param d: dict (name, value) i.e. ('loss', 0.11)
        :return:
        """
        for k,v in d.items():
            self.plot(k, v)

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        :param name: 图片名
        :param img_: N， C， W， H ！
        :param kwargs:
        :return:
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def img_many(self, d):
        """
        一次性放出多个img
        :param d:
        :return:
        """
        for k, v in d.items():
            self.img(k, v)

    def log(self, info, win="log_text"):
        """
        生成日志并显示
        self.log({'loss':1, 'lr':0.0001})
        :param info:
        :param win:
        :return:
        """
        self.log_text += ('[{time}]{info}<br>'.format(time=time.strftime('%m%d_%H%M%S'),
                                                      info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        自定义的plot,image,log,plot_many等除外，就是获取信息
        :param name:
        :return:
        """
        return getattr(self.vis, name)

