class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/pretrained_networks'
        self.got10k_val_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/got10k/val'
        self.lasot_lmdb_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/coco_lmdb'
        self.coco_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/coco'
        self.lasot_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/lasot'
        self.got10k_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/got10k/train'
        self.trackingnet_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/trackingnet'
        self.depthtrack_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/depthtrack/train'
        self.lasher_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/lasher/trainingset'
        self.visevent_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/visevent/train'
        self.luart_dir = '/DATA/wangyuhang/RGBT-Tracking/SFCATrack-main/data/LUART'
