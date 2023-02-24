from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394', '396']
    AIST_subjects = ['d16', 'd17', 'd18', 'd19', 'd20']

    if cfg.category == 'human_nerf' and cfg.task == 'zju_mocap':
        for sub in subjects:
            dataset_attrs.update({
                f"zju_{sub}_train": {
                    "dataset_path": f"dataset/zju_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"zju_{sub}_test": {
                    "dataset_path": f"dataset/zju_mocap/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap'
                },
            })


    if cfg.category == 'human_nerf' and cfg.task == 'smpl_mocap_plain':
        for sub in subjects:
            dataset_attrs.update({
                f"zju_{sub}_train": {
                    "dataset_path": f"dataset/smpl_mocap_plain/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"zju_{sub}_test": {
                    "dataset_path": f"dataset/smpl_mocap_plain/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'smpl_mocap_plain'
                },
            })


    if cfg.category == 'human_nerf' and cfg.task == 'smpl_mocap_color_v1':
        for sub in subjects:
            dataset_attrs.update({
                f"zju_{sub}_train": {
                    "dataset_path": f"dataset/smpl_mocap_color_v1/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"zju_{sub}_test": {
                    "dataset_path": f"dataset/smpl_mocap_color_v1/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'smpl_mocap_color_v1'
                },
            })


    if cfg.category == 'human_nerf' and cfg.task == 'wild':
        dataset_attrs.update({
            "monocular_train": {
                "dataset_path": 'dataset/wild/monocular',
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "monocular_test": {
                "dataset_path": 'dataset/wild/monocular',  
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'wild'
            },
        })


    if cfg.category == 'human_nerf' and cfg.task == 'AIST_mocap':
        for sub in AIST_subjects:
            dataset_attrs.update({
                f"AIST_{sub}_train": {
                    "dataset_path": f'dataset/AIST_mocap/{sub}',
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"AIST_{sub}_test": {
                    "dataset_path": f'dataset/AIST_mocap/{sub}',
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'wild'
                },
            })

    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
