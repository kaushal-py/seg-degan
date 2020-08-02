from types import SimpleNamespace
from code.classification.cifar10_system import Cifar10System

def main():

    # hparams = SimpleNamespace(
    #         batch_size = 128,
    #         lr = 0.1,
    #         lr_gamma = 0.1,
    #         lr_milestones = [80, 120],
    #         max_lr = 0.2,
    #         step_size_up = 50,
    #         lr_scheduler = 'step',
    #         epochs = 200,
    #         val_split = 0.02,
    #         )

    # config = SimpleNamespace(
    #         dataset_path = 'data/Cifar',
    #         model = 'resnet34',
    #         log_dir = 'logs/classification/cifar10/resnet34/v4',
    #         )

    # system = Cifar10System(config, hparams)
    # system.fit()

    hparams = SimpleNamespace(
            batch_size = 128,
            # lr = 0.1,
            # lr_gamma = 0.1,
            # lr_milestones = [80, 120],
            lr = 0,
            max_lr = 0.2,
            step_size_up = 100,
            lr_scheduler = 'cyclic',
            epochs = 200,
            )

    config = SimpleNamespace(
            dataset_path = 'data/Cifar',
            model = 'resnet34',
            log_dir = 'logs/classification/cifar10/resnet34/v12',
            )

    system = Cifar10System(config, hparams)
    system.fit()

    hparams = SimpleNamespace(
            batch_size = 128,
            # lr = 0.1,
            # lr_gamma = 0.1,
            # lr_milestones = [80, 120],
            lr = 0,
            max_lr = 0.1,
            step_size_up = 100,
            lr_scheduler = 'cyclic',
            epochs = 200,
            )

    config = SimpleNamespace(
            dataset_path = 'data/Cifar',
            model = 'resnet34',
            log_dir = 'logs/classification/cifar10/resnet34/v13',
            )

    system = Cifar10System(config, hparams)
    system.fit()

    # hparams = SimpleNamespace(
    #         batch_size = 128,
    #         # lr = 0.1,
    #         # lr_gamma = 0.1,
    #         # lr_milestones = [80, 120],
    #         lr = 0,
    #         max_lr = 0.2,
    #         step_size_up = 100,
    #         lr_scheduler = 'cyclic',
    #         epochs = 200,
    #         )

    # config = SimpleNamespace(
    #         dataset_path = 'data/Cifar',
    #         model = 'resnet18',
    #         log_dir = 'logs/classification/cifar10/resnet18/v8',
    #         )

    # system = Cifar10System(config, hparams)
    # system.fit()


def validate():

    hparams = SimpleNamespace(
            batch_size = 128,
            # lr = 0.1,
            # lr_gamma = 0.1,
            # lr_milestones = [80, 120],
            lr = 0,
            max_lr = 0.2,
            step_size_up = 100,
            lr_scheduler = 'cyclic',
            epochs = 200,
            )

    config = SimpleNamespace(
            dataset_path = 'data/Cifar',
            model = 'resnet34',
            log_dir = 'logs/classification/cifar10/resnet34/test/',
            )

    system = Cifar10System(config, hparams)
    system.load_from_checkpint('logs/classification/cifar10/resnet34/gaurav_model/best.tar')
    test_acc = system.test_epoch(0, split='Test')
    print("Test acc = {}".format(test_acc))



if __name__ == '__main__':
    # main()
    validate()
