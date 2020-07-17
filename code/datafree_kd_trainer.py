from types import SimpleNamespace
from code.datafree_kd_system import DatafreeKDSystem


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

    for version in range(3):
        config = SimpleNamespace(
                dataset = 'CamVid',
                dataset_path = 'data/CamVid',
                teacher = 'resnet50_pretrained',
                model = 'mobilenet',
                log_dir = 'logs/kd/datafree/mobilenet/v'+str(version+16),
                teacher_checkpoint = 'logs/segmentation/camvid/resnet50/v2/best.tar',
                generator_checkpoint = 'logs/gan/camvid/v1/epoch_500.tar',
                save_checkpoint = 'best',
                test_mode = 'val',
                )

        system = DatafreeKDSystem(config, hparams)
        system.fit()

if __name__ == '__main__':
    main()
