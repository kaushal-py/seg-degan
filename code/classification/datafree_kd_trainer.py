from types import SimpleNamespace
from code.classification.datafree_kd_system import DatafreeKDSystem

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
            generator_checkpoint = 'logs/classification/gan/resnet/svhn/v3/epoch_200.tar',
            alpha = 1,
            temperature = 20,
            nz = 100,
            # cube_size = 2,
            )

    config = SimpleNamespace(
            dataset_path = 'data/Cifar',
            model = 'resnet18',
            teacher = 'resnet34',
            log_dir = 'logs/classification/datafree_kd/svhn/50_500'
            )

    system = DatafreeKDSystem(config, hparams)
    system.fit()

    # hparams = SimpleNamespace(
    #         batch_size = 128,
    #         batch_length = 50000//128,
    #         lr = 0.0001,
    #         max_lr = 0.2,
    #         step_size_up = 50,
    #         step_size_down = 150,
    #         # lr = 0.001,
    #         # lr_milestones = [100],
    #         # lr_gamma = 0.1,
    #         lr_scheduler = 'cyclic',
    #         epochs = 200,
    #         teacher_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
    #         generator_checkpoint = 'logs/classification/gan/resnet/svhn/v2/epoch_200.tar',
    #         alpha = 1,
    #         temperature = 20,
    #         nz = 100,
    #         # cube_size=10,
    #         )

    # config = SimpleNamespace(
    #         dataset_path = 'data/Cifar',
    #         model = 'resnet18',
    #         teacher = 'resnet34',
    #         log_dir = 'logs/classification/datafree_kd/svhn/0_30'
    #         )

    # system = DatafreeKDSystem(config, hparams)
    # system.fit()

def multiple_run():

    # for proxy in ['svhn', 'cifar100_40', 'cifar100_90']:
    # for gan in ['gan', 'degan_0_30']:
    for gan in ['_2048']:
        # for gan in ['sravanti_gan', 'sravanti_degan']:
        for run in ['0_5', '0_10']:
        # for run in ['0_10']:
        # for run in [1, 2, 3]:
        # for run in [64, 2048]:
        # for gan in ['v8']:
            hparams = SimpleNamespace(
                    batch_size = 128,
                    batch_length = 50000//128,
                    lr = 0.0001,
                    max_lr = 0.2,
                    step_size_up = 50,
                    step_size_down = 150,
                    # lr_milestones = [200],
                    # lr_gamma = 0.1,
                    # lr = 0.001,
                    lr_scheduler = 'cyclic',
                    epochs = 200,
                    # teacher_checkpoint = 'logs/classification/cifar11/resnet34/gaurav_model/best.tar',
                    teacher_checkpoint = 'logs/classification/cifar10/alexnet/final/best_model.pth',
                    # generator_checkpoint = 'logs/classification/gan/alexnet/cifar100_household/'+gan+'_'+str(run)+'/epoch_200.tar',
                    generator_checkpoint = 'logs/classification/gan/alexnet/new_cifar100_household/gan_'+run+str(gan)+'/epoch_200.tar',
                    alpha = 1,
                    temperature = 20,
                    nz = 100,
                    )

            config = SimpleNamespace(
                    dataset_path = 'data/Cifar',
                    model = 'alexnet_half',
                    teacher = 'alexnet',
                    log_dir = 'logs/classification/datafree_kd/alexnet/new_cifar100_household/gan_cyclic_'+run+str(gan),
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

def validate():

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
            generator_checkpoint = 'logs/classification/gan/cifar100_90/v1/epoch_200.tar',
            alpha = 0.9,
            temperature = 20,
            nz = 100,
            )

    config = SimpleNamespace(
            dataset_path = 'data/Cifar',
            model = 'alexnet_half',
            teacher = 'alexnet',
            log_dir = 'logs/classification/datafree_kd/test'
            )

    system = DatafreeKDSystem(config, hparams)
    system.load_from_checkpint('logs/classification/datafree_kd/alexnet_half_cifar100/v8/best.tar')
    # system.fit()


if __name__ == '__main__':
    # main()
    # validate()
    multiple_run()
    # epoch_tuning()
    # temp_tuning()
    # alpha_tuning()
