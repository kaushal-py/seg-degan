from types import SimpleNamespace
from code.degan_joint_system import DeGanJointSystem

def main():

    hparams = SimpleNamespace(
            train_batch_size = 64,
            test_batch_size = 16,
            lr = 0.0002,
            num_epochs = 100,
            entropy_weight = 0,
            diversity_weight = 0,
            diversity = 'entropy',
            nz = 100,
            model_checkpoint = 'logs/segmentation/camvid/resnet50/v2/best.tar',
            momentum = 0.9,
            weight_decay = 5e-4,
            s_lr = 0.1,
            lr_scheduler = True,
            scheduler_step = 100,
            scheduler_gamma = 0.1,
            )

    config = SimpleNamespace(
            dataset = 'Cityscapes',
            proxy_path = 'data/Cityscapes',
            dataset_path = 'data/CamVid',
            model = 'resnet50_pretrained',
            student = 'mobilenet',
            log_dir = 'logs/joint/camvid/v3',
            save_checkpoint = 'best',
            test_mode = 'val',
            num_classes = 11,
            )

    system = DeGanJointSystem(config, hparams)
    system.fit()


def hparam_tuning():

    for i, (ent, div) in enumerate([(0,0), (0, 10), (10, 0), (5, 5), (5, 10)]):
        hparams = SimpleNamespace(
                train_batch_size = 64,
                test_batch_size = 8,
                lr = 0.0002,
                num_epochs = 100,
                entropy_weight = ent,
                diversity_weight = div,
                diversity = 'entropy',
                nz = 100,
                model_checkpoint = 'logs/segmentation/camvid/resnet50/v2/best.tar',
                )

        config = SimpleNamespace(
                dataset = 'Cityscapes',
                dataset_path = 'data/Cityscapes',
                model = 'resnet50_pretrained',
                log_dir = 'logs/gan/cityscapes/v'+str(i+1),
                save_checkpoint = 'best',
                test_mode = 'test',
                num_classes = 11,
                checkpoint_interval = 20,
                )

        system = DeGanSystem(config, hparams)
        system.fit()


if __name__ == '__main__':
    main()
    # hparam_tuning()
