import warnings


class DefaultConfig(object):
    env = 'default'
    model = 'ResNet34'

    train_data_root = './data/train/'
    test_data_root = './data/test1'
    # load_model_path = 'checkpoints/model.pth'
    load_model_path = None

    batch_size = 8
    use_gpu = False
    num_workers = 1
    print_feq = 1

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
