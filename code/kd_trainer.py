from types import SimpleNamespace
from code.kd_system import KDSystem


def main():

    hparams = SimpleNamespace(
            train_batch_size = 16,
            test_batch_size = 16,
            momentum = 0.9,
            weight_decay = 5e-4,
            lr = 0.1,
            lr_scheduler = True,
            scheduler_step = 100,
            scheduler_gamma = 0.1,
            num_epochs = 300,
            kd_weight = 1,
            )

    config = SimpleNamespace(
            dataset = 'CamVid',
            dataset_path = 'data/CamVid',
            teacher = 'resnet50_pretrained',
            model = 'mobilenet',
            log_dir = 'logs/datadriven/mobilenet/v5',
            teacher_checkpoint = 'logs/segmentation/resnet50/v9/best.tar',
            save_checkpoint = 'best',
            )

    system = KDSystem(config, hparams)
    system.fit()

    config = SimpleNamespace(
            dataset = 'CamVid',
            dataset_path = 'data/CamVid',
            teacher = 'resnet50_pretrained',
            model = 'mobilenet',
            log_dir = 'logs/datadriven/mobilenet/v6',
            teacher_checkpoint = 'logs/segmentation/resnet50/v9/best.tar',
            save_checkpoint = 'best',
            )

    system = KDSystem(config, hparams)
    system.fit()

if __name__ == '__main__':
    main()
