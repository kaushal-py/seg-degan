from types import SimpleNamespace
from code.kd_system import KDSystem


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
            kd_weight = 1,
            )

    config = SimpleNamespace(
            dataset = 'Nyu',
            dataset_path = 'data/Nyu',
            teacher = 'resnet50_pretrained',
            model = 'mobilenet',
            log_dir = 'logs/kd/datadriven/mobilenet/v1',
            teacher_checkpoint = 'logs/segmentation/256/resnet50/v1/best.tar',
            save_checkpoint = 'best',
            test_mode = 'val',
            )

    system = KDSystem(config, hparams)
    system.fit()


if __name__ == '__main__':
    main()
