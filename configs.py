
def build_config(dataset):
    cfg = type('', (), {})()
    if dataset in ['ucf', 'ucf-crime']:
        cfg.dataset = 'ucf-crime'
        cfg.model_name = 'ucf_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = './data/pyj/feat/ucf-i3d'
        cfg.train_list = './list/ucf/train.list'
        cfg.test_list = './list/ucf/test.list'
        cfg.token_feat = './list/ucf/ucf-prompt.npy'
        cfg.gt = './list/70_15_15/ucf-gt.npy'    #adjust the path
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        # CC settings
        cfg.t_step = 9
        # training settings
        cfg.temp = 0.09
        cfg.lamda = 1
        cfg.seed = 9
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'slide'  # ['fixed': 10, slide': 7]
        cfg.kappa = 7  # smooth window
        cfg.ckpt_path = './ckpt/ucf__8636.pkl'

    # base settings
    cfg.feat_dim = 1024
    cfg.head_num = 1
    cfg.hid_dim = 128
    cfg.out_dim = 300
    cfg.lr = 5e-4
    cfg.dropout = 0.1
    cfg.train_bs = 128
    cfg.max_seqlen = 200
    cfg.max_epoch = 50
    cfg.workers = 8
    cfg.save_dir = './ckpt/'
    cfg.logs_dir = './log_info.log'

    #yolov9 settings
    cfg.anomaly_threshold = 0.5     #threshold for yolo infer anomaly frames
    cfg.weights = './ckpt/70_15_15/last.pt'    #adjust the path
    cfg.batch_size = 64
    cfg.imgsz = 256
    cfg.conf_thres = 0.001
    cfg.iou_thres = 0.7
    cfg.max_det = 300
    cfg.workers = 8
    cfg.single_cls = True
    cfg.save_json = True
    cfg.project = './runs/val'
    cfg.name = 'exp'
    cfg.exist_ok = True
    cfg.data = './datasets/frame_ucfcrime_70_15_15/test.yaml'    #adjust the path
    
    return cfg
