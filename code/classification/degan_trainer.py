from types import SimpleNamespace
from code.classification.degan_system import DeGanSystem

def main():

    hparams = SimpleNamespace(
            batch_size = 2048,
            lr = 0.0002,
            num_epochs = 200,
            entropy_weight = 0,
            diversity_weight = 20,
            diversity = 'entropy',
            nz = 100,
            model_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
            )

    config = SimpleNamespace(
            dataset = 'cifar100',
            num_classes = 10,
            dataset_path = 'data/Cifar',
            model = 'resnet34',
            log_dir = 'logs/classification/gan/resnet/cifar100_90/v8',
            checkpoint_interval = 50,
            )

    system = DeGanSystem(config, hparams)
    system.fit()

    hparams = SimpleNamespace(
            batch_size = 2048,
            lr = 0.0002,
            num_epochs = 200,
            entropy_weight = 0,
            diversity_weight = 15,
            diversity = 'entropy',
            nz = 100,
            model_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
            )

    config = SimpleNamespace(
            dataset = 'cifar100',
            num_classes = 10,
            dataset_path = 'data/Cifar',
            model = 'resnet34',
            log_dir = 'logs/classification/gan/resnet/cifar100_90/v9',
            checkpoint_interval = 50,
            )

    system = DeGanSystem(config, hparams)
    system.fit()

def hparam_tuning():

    for i, (ent, div) in enumerate([(0, 30), (0, 40), (0, 25)]):
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
                dataset = 'cifar100',
                num_classes = 10,
                dataset_path = 'data/Cifar',
                model = 'resnet34',
                log_dir = 'logs/classification/gan/resnet/cifar100_90/v'+str(i+10),
                checkpoint_interval = 50,
                )

        system = DeGanSystem(config, hparams)
        system.fit()


if __name__ == '__main__':
    # main()
    hparam_tuning()
