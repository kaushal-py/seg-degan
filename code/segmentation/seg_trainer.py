from types import SimpleNamespace
from code.segmentation.seg_system import SegmentationSystem


def main():

    hparams = SimpleNamespace(
            train_batch_size = 64,
            test_batch_size = 16,
            momentum = 0.9,
            weight_decay = 5e-4,
            lr = 0.05,
            lr_scheduler = True,
            scheduler_step = 100,
            scheduler_gamma = 0.1,
            num_epochs = 300,
            )

    config = SimpleNamespace(
            dataset = 'Nyu',
            dataset_path = 'data/Nyu',
            model = 'resnet50_pretrained',
            log_dir = 'logs/segmentation/256/resnet50/v1',
            save_checkpoint = 'best',
            test_mode = 'val',
            )

    system = SegmentationSystem(config, hparams)
    system.fit()


def validate():

    hparams = SimpleNamespace(
            train_batch_size = 64,
            test_batch_size = 8,
            momentum = 0.9,
            weight_decay = 5e-4,
            lr = 0.05,
            lr_scheduler = True,
            scheduler_step = 100,
            scheduler_gamma = 0.1,
            num_epochs = 300,
            )

    config = SimpleNamespace(
            dataset = 'Nyu',
            dataset_path = 'data/Nyu',
            model = 'mobilenet',
            log_dir = 'logs/segmentation/validate',
            save_checkpoint = 'best',
            test_mode = 'val',
            )

    system = SegmentationSystem(config, hparams)
    system.load_from_checkpint('logs/kd/datafree/mobilenet/v1/best.tar')
    system.test_epoch(0)


if __name__ == '__main__':
    # main()
    validate()
