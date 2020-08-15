from types import SimpleNamespace
from code.classification.degan_system import DeGanSystem

def main():

    # hparams = SimpleNamespace(
    #         batch_size = 2048,
    #         lr = 0.0002,
    #         num_epochs = 200,
    #         entropy_weight = 0,
    #         diversity_weight = 0,
    #         diversity = 'entropy',
    #         nz = 100,
    #         model_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
    #         )

    # config = SimpleNamespace(
    #         dataset = 'svhn',
    #         num_classes = 10,
    #         dataset_path = 'data/Svhn',
    #         model = 'resnet34',
    #         log_dir = 'logs/classification/gan/resnet/svhn/v1',
    #         checkpoint_interval = 50,
    #         )

    # system = DeGanSystem(config, hparams)
    # system.fit()

    hparams = SimpleNamespace(
            batch_size = 64,
            lr = 0.0002,
            num_epochs = 200,
            entropy_weight = 50,
            diversity_weight = 500,
            diversity = 'entropy',
            nz = 100,
            # model_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
            model_checkpoint = 'logs/classification/cifar10/alexnet/final/best_model.pth',
            )

    config = SimpleNamespace(
            dataset = 'svhn',
            num_classes = 10,
            dataset_path = 'data/Svhn',
            model = 'alexnet',
            log_dir = 'logs/classification/gan/alexnet/svhn/v7',
            checkpoint_interval = 50,
            )

    system = DeGanSystem(config, hparams)
    system.fit()

    hparams = SimpleNamespace(
            batch_size = 2048,
            lr = 0.0002,
            num_epochs = 200,
            entropy_weight = 50,
            diversity_weight = 500,
            diversity = 'entropy',
            nz = 100,
            # model_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
            model_checkpoint = 'logs/classification/cifar10/alexnet/final/best_model.pth',
            )

    config = SimpleNamespace(
            dataset = 'svhn',
            num_classes = 10,
            dataset_path = 'data/Svhn',
            model = 'alexnet',
            log_dir = 'logs/classification/gan/alexnet/svhn/v8',
            checkpoint_interval = 50,
            )

    system = DeGanSystem(config, hparams)
    system.fit()

def validate():

    hparams = SimpleNamespace(
            batch_size = 2048,
            lr = 0.0002,
            num_epochs = 200,
            entropy_weight = 0,
            diversity_weight = 0,
            diversity = 'entropy',
            nz = 100,
            model_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
            )

    config = SimpleNamespace(
            dataset = 'svhn',
            num_classes = 10,
            dataset_path = 'data/Svhn',
            model = 'resnet34',
            log_dir = 'logs/classification/gan/resnet/test',
            checkpoint_interval = 50,
            )

    system = DeGanSystem(config, hparams)
    system.load_from_checkpint('logs/classification/gan/resnet/svhn/v3/epoch_200.tar')
    system.entropy_and_diversity(batches=10)

def hparam_tuning():

    for i, (ent, div) in enumerate([(0, 0), (0, 5), (0, 30)]):
        hparams = SimpleNamespace(
                batch_size = 2048,
                lr = 0.0002,
                num_epochs = 200,
                entropy_weight = ent,
                diversity_weight = div,
                diversity = 'entropy',
                nz = 100,
                # model_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
                model_checkpoint = 'logs/classification/cifar10/alexnet/final/best_model.pth',
                )

        config = SimpleNamespace(
                dataset = 'cifar100',
                num_classes = 10,
                dataset_path = 'data/Cifar',
                model = 'alexnet',
                log_dir = 'logs/classification/gan/alexnet/cifar100_40/v'+str(i+1),
                checkpoint_interval = 50,
                )

        system = DeGanSystem(config, hparams)
        system.fit()

def hparam_tuning_2():

    for i, (ent, div) in enumerate([(100, 500), (200, 500)]):
        hparams = SimpleNamespace(
                batch_size = 2048,
                lr = 0.0002,
                num_epochs = 200,
                entropy_weight = ent,
                diversity_weight = div,
                diversity = 'entropy',
                nz = 100,
                model_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
                )

        config = SimpleNamespace(
                dataset = 'svhn',
                num_classes = 10,
                dataset_path = 'data/Svhn',
                model = 'resnet34',
                log_dir = 'logs/classification/gan/resnet/svhn/v'+str(i+4),
                checkpoint_interval = 50,
                )

        system = DeGanSystem(config, hparams)
        system.fit()

if __name__ == '__main__':
    main()
    # validate()
    # hparam_tuning()
