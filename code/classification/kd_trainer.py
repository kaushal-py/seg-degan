from types import SimpleNamespace
from code.classification.kd_system import KDSystem

def main():

    hparams = SimpleNamespace(
            batch_size = 128,
            lr = 0,
            max_lr = 0.2,
            step_size_up = 50,
            step_size_down = 150,
            lr_scheduler = 'cyclic',
            epochs = 200,
            teacher_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
            alpha = 1,
            temperature = 20,
            )

    config = SimpleNamespace(
            dataset_path = 'data/Cifar',
            model = 'resnet18',
            teacher = 'resnet34',
            log_dir = 'logs/classification/datdriven_kd/resnet18/cifar100_one_from_each/v1',
           )

    system = KDSystem(config, hparams)
    system.fit()

def temp_tuning():

    for idx, temp in enumerate([2, 4, 5, 10, 15, 20, 30]):
        hparams = SimpleNamespace( batch_size = 64,
                lr = 0.1,
                lr_gamma = 0.1,
                lr_step_size = 80,
                max_lr = 0.2,
                lr_scheduler = True,
                epochs = 200,
                val_split = 0.2,
                teacher_checkpoint = 'logs/classification/cifar10/alexnet/v16/best.tar',
                alpha = 0.5,
                temperature = temp,
                )

        config = SimpleNamespace(
                dataset_path = 'data/Cifar',
                model = 'alexnet_half',
                teacher = 'alexnet',
                log_dir = 'logs/classification/datdriven_kd/alexnet_half/v'+str(idx+2),
                )

        system = KDSystem(config, hparams)
        system.fit()

def alpha_tuning():

    for idx, a in enumerate([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]):
        hparams = SimpleNamespace( batch_size = 64,
                lr = 0.1,
                lr_gamma = 0.1,
                lr_step_size = 80,
                max_lr = 0.2,
                lr_scheduler = True,
                epochs = 200,
                val_split = 0.2,
                teacher_checkpoint = 'logs/classification/cifar10/alexnet/v16/best.tar',
                alpha = a,
                temperature = 4,
                )

        config = SimpleNamespace(
                dataset_path = 'data/Cifar',
                model = 'alexnet_half',
                teacher = 'alexnet',
                log_dir = 'logs/classification/datdriven_kd/alexnet_half/v'+str(idx+23),
                )

        system = KDSystem(config, hparams)
        system.fit()

if __name__ == '__main__':
    main()
    # temp_tuning()
    # alpha_tuning()
