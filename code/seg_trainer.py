from types import SimpleNamespace
from code.seg_system import SegmentationSystem


def main():

    config = SimpleNamespace(
            dataset = 'CamVid',
            dataset_path = 'data/CamVid',
            model = 'resnet50_pretrained',
            log_dir = 'logs/trial',
            save_checkpoint = 'best',
            )
    hparams = SimpleNamespace(
            train_batch_size = 16,
            test_batch_size = 8,
            momentum = 0.9,
            weight_decay = 5e-4,
            lr = 0.1,
            lr_scheduler = True,
            scheduler_step = 100,
            scheduler_gamma = 0.1,
            num_epochs = 300,
            )

    system = SegmentationSystem(config, hparams)
    system.fit()

if __name__ == '__main__':
    main()
