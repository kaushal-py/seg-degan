from types import SimpleNamespace
from code.classification.degan_system import DeGanSystem

def main():

    hparams = SimpleNamespace(
            batch_size = 2048,
            lr = 0.0002,
            num_epochs = 200,
            entropy_weight = 0,
            diversity_weight = 5,
            diversity = 'entropy',
            nz = 100,
            model_checkpoint = 'logs/classification/cifar10/alexnet/v16/best.tar',
            )

    config = SimpleNamespace(
            dataset = 'cifar100',
            num_classes = 10,
            dataset_path = 'data/Cifar',
            model = 'alexnet',
            log_dir = 'logs/classification/gan/cifar100_90/v2',
            checkpoint_interval = 20,
            )

    system = DeGanSystem(config, hparams)
    system.fit()


def hparam_tuning():

    for i, (ent, div) in enumerate([(0, 10), (10, 0), (5, 5), (10, 10), (5, 10)]):
        hparams = SimpleNamespace(
                batch_size = 2048,
                lr = 0.0002,
                num_epochs = 200,
                entropy_weight = ent,
                diversity_weight = div,
                diversity = 'entropy',
                nz = 100,
                model_checkpoint = 'logs/classification/cifar10/alexnet/v16/best.tar',
                )

        config = SimpleNamespace(
                dataset = 'cifar100',
                num_classes = 10,
                dataset_path = 'data/Cifar',
                model = 'alexnet',
                log_dir = 'logs/classification/gan/cifar100/v'+str(i+3),
                checkpoint_interval = 50,
                )

        system = DeGanSystem(config, hparams)
        system.fit()


if __name__ == '__main__':
    main()
    # hparam_tuning()
