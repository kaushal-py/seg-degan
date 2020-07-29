from types import SimpleNamespace
from code.classification.cifar10_system import Cifar10System

def main():

    hparams = SimpleNamespace(
            batch_size = 64,
            lr = 0.001,
            lr_gamma = 0.1,
            lr_step_size = 80,
            max_lr = 0.2,
            step_size_up = 50,
            lr_scheduler = True,
            epochs = 200,
            val_split = 0.2,
            )

    config = SimpleNamespace(
            dataset_path = 'data/Cifar',
            model = 'alexnet',
            log_dir = 'logs/classification/cifar10/alexnet/v24',
            )

    system = Cifar10System(config, hparams)
    system.fit()

if __name__ == '__main__':
    main()
