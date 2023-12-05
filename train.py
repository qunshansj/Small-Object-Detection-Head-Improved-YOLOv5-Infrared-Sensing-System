python



class YOLOv5Trainer:
    def __init__(self, hyp, opt, device):
        self.hyp = hyp
        self.opt = opt
        self.device = device

    def train(self):
        save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, = \
            self.opt.save_dir, self.opt.epochs, self.opt.batch_size, self.opt.weights, self.opt.single_cls, \
            self.opt.evolve, self.opt.data, self.opt.cfg, self.opt.resume, self.opt.noval, self.opt.nosave, \
            self.opt.workers

        # Directories
        save_dir = Path(save_dir)
        wdir = save_dir / 'weights'
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        last = wdir / 'last.pt'
        best = wdir / 'best.pt'
        results_file = save_dir / 'results.txt'

        # Hyperparameters
        if isinstance(self.hyp, str):
            with open(self.hyp) as f:
                self.hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in self.hyp.items()))

        # Save run settings
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(self.hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(self.opt), f, sort_keys=False)

        # Configure
        plots = not evolve  # create plots
        cuda = self.device.type != 'cpu'
        init_seeds(1 + RANK)
        with open(data) as f:
            data_dict = yaml.safe_load(f)  # data dict

        # Loggers
        loggers = {'wandb': None, 'tb': None}  # loggers dict
        if RANK in [-1, 0]:
            # TensorBoard
            if not evolve:
                prefix = colorstr('tensorboard: ')
                LOGGER.info(f"{prefix}Start with 'tensorboard --logdir {self.opt.project}', view at http://localhost:6006/")
                loggers['tb'] = SummaryWriter(str(save_dir))

            # W&B
            self.opt.hyp = self.hyp  # add hyperparameters
            run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
            run_id = run_id if self.opt.resume else None  # start fresh run if transfer learning
            wandb_logger = WandbLogger(self.opt, save_dir.stem, run_id, data_dict)
            loggers['wandb'] = wandb_logger.wandb
            if loggers['wandb']:
                data_dict = wandb_logger.data_dict
                weights, epochs, self.hyp = self.opt.weights, self.opt.epochs, self.opt.hyp  # may update weights, epochs if resuming

        nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
        names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check
        is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

        # Model
        pretrained = weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(RANK):
                weights = attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location=self.device)  # load checkpoint
            model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=self.hyp.get('anchors')).to(self.device)  # create
            exclude = ['anchor'] if (cfg or self.hyp.get('anchors')) and not resume else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(state_dict, strict=False)  # load
            LOGGER.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
        else:
            model = Model(cfg, ch=3, nc=nc, anchors=self.hyp.get('anchors')).to(self.device)  # create
        with torch_distributed_zero_first(RANK):
            check_dataset(data_dict)  # check
        train_path = data_dict['train']
        val_path = data_dict['val']

        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        LOGGER.info(f"Scaled weight_decay = {self.hyp['weight_decay']}")

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        if self.opt.adam:
            optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': pg1,
