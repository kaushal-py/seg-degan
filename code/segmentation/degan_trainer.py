from types import SimpleNamespace
from code.segmentation.degan_system import DeGanSystem

def main():

    hparams = SimpleNamespace(
            train_batch_size = 64,
            test_batch_size = 16,
            lr = 0.0002,
            num_epochs = 20000,
            entropy_weight = 0,
            diversity_weight = 0,
            diversity = 'entropy',
            nz = 100,
            model_checkpoint = 'logs/segmentation/256/resnet50/v1/best.tar',
            )

    config = SimpleNamespace(
            dataset = 'Nyu',
            dataset_path = 'data/Nyu',
            model = 'resnet50_pretrained',
            log_dir = 'logs/gan/nyu_long/v3',
            save_checkpoint = 'best',
            test_mode = 'val',
            num_classes = 13,
            checkpoint_interval = 1000,
            )

    system = DeGanSystem(config, hparams)
    system.fit()


def hparam_tuning():

    for i, (ent, div) in enumerate([(0, 10), (10, 0), (5, 5), (10, 10), (5, 10)]):
        hparams = SimpleNamespace(
                train_batch_size = 32,
                test_batch_size = 16,
                lr = 0.0002,
                num_epochs = 500,
                entropy_weight = ent,
                diversity_weight = div,
                diversity = 'regression',
                nz = 100,
                model_checkpoint = 'logs/segmentation/256/resnet50/v1/best.tar',
                )

        config = SimpleNamespace(
                dataset = 'Nyu',
                dataset_path = 'data/Nyu',
                model = 'resnet50_pretrained',
                log_dir = 'logs/gan/nyu/v'+str(i+7),
                save_checkpoint = 'best',
                test_mode = 'val',
                num_classes = 13,
                checkpoint_interval = 100,
                )

        system = DeGanSystem(config, hparams)
        system.fit()


if __name__ == '__main__':
    main()
    # hparam_tuning()
