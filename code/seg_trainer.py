from types import SimpleNamespace
from code.seg_system import SegmentationSystem


def main():

    # hparams = SimpleNamespace(
    #         train_batch_size = 32,
    #         test_batch_size = 32,
    #         momentum = 0.9,
    #         weight_decay = 5e-4,
    #         lr = 0.001,
    #         lr_scheduler = True,
    #         scheduler_step = 40,
    #         scheduler_gamma = 0.5,
    #         num_epochs = 120,
    #         )

    hparams = SimpleNamespace(
            train_batch_size = 16,
            test_batch_size = 32,
            momentum = 0.9,
            weight_decay = 5e-4,
            lr = 0.1,
            lr_scheduler = True,
            scheduler_step = 100,
            scheduler_gamma = 0.1,
            num_epochs = 300,
            )

    config = SimpleNamespace(
            dataset = 'CamVid',
            dataset_path = 'data/CamVid',
            model = 'mobilenet',
            log_dir = 'logs/segmentation/mobilenet/v4',
            save_checkpoint = 'best',
            )

    system = SegmentationSystem(config, hparams)
    system.fit()

    config = SimpleNamespace(
            dataset = 'CamVid',
            dataset_path = 'data/CamVid',
            model = 'mobilenet',
            log_dir = 'logs/segmentation/mobilenet/v5',
            save_checkpoint = 'best',
            )

    system = SegmentationSystem(config, hparams)
    system.fit()

def validate():

    hparams = SimpleNamespace(
            train_batch_size = 64,
            test_batch_size = 4,
            momentum = 0.9,
            weight_decay = 5e-4,
            lr = 0.1,
            lr_scheduler = True,
            scheduler_step = 100,
            scheduler_gamma = 0.1,
            num_epochs = 300,
            )

    config = SimpleNamespace(
            dataset = 'CamVid',
            dataset_path = 'data/CamVid',
            model = 'resnet50_pretrained',
            log_dir = 'logs/segmentation/resnet50/validate',
            save_checkpoint = 'best',
            )

    system = SegmentationSystem(config, hparams)
    system.load_from_checkpint('logs/segmentation/resnet50/v6/best.tar')
    system.test_epoch(0)


if __name__ == '__main__':
    main()
    # validate()
