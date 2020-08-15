from types import SimpleNamespace
from code.classification.datafree_kd_system_2 import DatafreeKDSystem

def main():

    hparams = SimpleNamespace(
            batch_size = 128,
            batch_length = 50000//128,
            # batch_length = 25000//128,
            lr = 0.0001,
            max_lr = 0.2,
            step_size_up = 50,
            step_size_down = 150,
            # lr = 0.001,
            # lr_milestones = [100],
            # lr_gamma = 0.1,
            lr_scheduler = 'cyclic',
            epochs = 200,
            teacher_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
            generator_checkpoint = 'logs/classification/gan/resnet/synthia/sravanti/netG_epoch_199.pth',
            alpha = 1,
            temperature = 20,
            nz = 100,
            # cube_size = 2,
            )

    config = SimpleNamespace(
            dataset_path = 'data/Cifar',
            model = 'resnet18',
            teacher = 'resnet34',
            log_dir = 'logs/classification/datafree_kd/synthia/0_0'
            )

    system = DatafreeKDSystem(config, hparams)
    system.fit()

    hparams = SimpleNamespace(
            batch_size = 128,
            batch_length = 50000//128,
            # batch_length = 25000//128,
            lr = 0.0001,
            max_lr = 0.2,
            step_size_up = 50,
            step_size_down = 150,
            # lr = 0.001,
            # lr_milestones = [100],
            # lr_gamma = 0.1,
            lr_scheduler = 'cyclic',
            epochs = 200,
            teacher_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
            generator_checkpoint = 'logs/classification/gan/resnet/synthia/sravanti_degan/netG_epoch_199.pth',
            alpha = 1,
            temperature = 20,
            nz = 100,
            # cube_size = 2,
            )

    config = SimpleNamespace(
            dataset_path = 'data/Cifar',
            model = 'resnet18',
            teacher = 'resnet34',
            log_dir = 'logs/classification/datafree_kd/synthia/0_20'
            )

    system = DatafreeKDSystem(config, hparams)
    system.fit()

def multiple_run():

    for version in (3, 7):
        hparams = SimpleNamespace(
                batch_size = 128,
                batch_length = 50000//128,
                lr = 0.0001,
                max_lr = 0.2,
                step_size_up = 50,
                step_size_down = 150,
                lr_scheduler = 'cyclic',
                epochs = 200,
                teacher_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
                generator_checkpoint = 'logs/classification/gan/resnet/cifar100_6/v'+str(version)+'/epoch_200.tar',
                alpha = 1,
                temperature = 20,
                nz = 100,
                )

        config = SimpleNamespace(
                dataset_path = 'data/Cifar',
                model = 'resnet18',
                teacher = 'resnet34',
                log_dir = 'logs/classification/datafree_kd/resnet18_cifar100_6/v'+str(version),
                )

        system = DatafreeKDSystem(config, hparams)
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
                nz = 100,
                )

        config = SimpleNamespace(
                dataset_path = 'data/Cifar',
                model = 'alexnet_half',
                teacher = 'alexnet',
                log_dir = 'logs/classification/datdriven_kd/alexnet_half/v'+str(idx+2),
                )

        system = DatafreeKDSystem(config, hparams)
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

        system = DatafreeKDSystem(config, hparams)
        system.fit()

def epoch_tuning():

    for idx, epoch in enumerate([20, 40, 60, 80, 100, 120, 140, 160, 180]):

        hparams = SimpleNamespace(
                batch_size = 64,
                batch_length = 625,
                lr = 0.1,
                lr_gamma = 0.1,
                lr_step_size = 80,
                max_lr = 0.2,
                lr_scheduler = True,
                epochs = 200,
                val_split = 0.2,
                teacher_checkpoint = 'logs/classification/cifar10/alexnet/v16/best.tar',
                generator_checkpoint = 'logs/classification/gan/cifar100_90/v1/epoch_'+str(epoch)+'.tar',
                alpha = 0.9,
                temperature = 20,
                nz = 100,
                )

        config = SimpleNamespace(
                dataset_path = 'data/Cifar',
                model = 'alexnet_half',
                teacher = 'alexnet',
                log_dir = 'logs/classification/datafree_kd/alexnet_half_cifar100_90_epoch/v'+str(idx),
                )

        system = DatafreeKDSystem(config, hparams)
        system.fit()

if __name__ == '__main__':
    main()
    # multiple_run()
    # epoch_tuning()
    # temp_tuning()
    # alpha_tuning()