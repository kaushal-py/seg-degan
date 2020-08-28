import shutil
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
            entropy_weight = 0,
            diversity_weight = 50,
            diversity = 'regression',
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


def validate():

    #for i, (ent, div) in enumerate([(50, 1000),(50, 1500),(50, 2000),(0, 500),(100, 500),(200, 500),(500, 500),(100, 1000)]):
    for i, (ent, div) in enumerate([(0, 0)]):

        hparams = SimpleNamespace(
                # batch_size = batch_size,
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
                dataset = 'svhn',
                num_classes = 10,
                dataset_path = 'data/Svhn',
                model = 'alexnet',
                log_dir = 'logs/classification/gan/alexnet/svhn/test',
                checkpoint_interval = 50,
                )

        system = DeGanSystem(config, hparams)
        # system.load_from_checkpint('logs/classification/gan/alexnet/svhn/degan_2048_'+str(ent)+'_'+str(div)+'/epoch_200.tar')
        system.load_from_checkpint('logs/classification/gan/alexnet/svhn/gan_batchsize_2048/epoch_200.tar')
        print("Hparams: {}, {}:".format(ent, div))
        system.entropy_and_diversity(batches=10)
        shutil.rmtree('logs/classification/gan/alexnet/svhn/test')

def hparam_tuning():

    for i, (ent, div) in enumerate([(0, 50), (0, 100)]):

        hparams = SimpleNamespace(
                # batch_size = batch_size,
                batch_size = 2048,
                lr = 0.0002,
                num_epochs = 200,
                entropy_weight = ent,
                diversity_weight = div,
                diversity = 'regression',
                nz = 100,
                # model_checkpoint = 'logs/classification/cifar10/resnet34/gaurav_model/best.tar',
                model_checkpoint = 'logs/classification/cifar10/alexnet/final/best_model.pth',
                )

        config = SimpleNamespace(
                dataset = 'svhn',
                num_classes = 10,
                dataset_path = 'data/Svhn',
                model = 'alexnet',
                log_dir = 'logs/classification/gan/alexnet/svhn/degan_regression'+str(ent)+'_'+str(div),
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

def batchsize_tuning():
    for ent, div in [(0,1)]:
        for batch_size in [2048]:
        # for batch_size in [1, 2, 3]:
            hparams = SimpleNamespace(
                    batch_size = batch_size,
                    # batch_size = 2048,
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
                    log_dir = 'logs/classification/gan/alexnet/new_cifar100_household/gan_'+str(ent)+'_'+str(div)+'_'+str(batch_size),
                    checkpoint_interval = 50,
                    )

            system = DeGanSystem(config, hparams)
            system.fit()

if __name__ == '__main__':
    # main()
    batchsize_tuning()
    # validate()
    # hparam_tuning()
