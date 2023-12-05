python



class ModelExporter:
    def __init__(self, weights='./yolov5s.pt', img_size=(640, 640), batch_size=1, device='cpu',
                 include=('torchscript', 'onnx', 'coreml'), half=False, inplace=False, train=False,
                 optimize=False, dynamic=False, simplify=False, opset_version=12):
        self.weights = weights
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device
        self.include = include
        self.half = half
        self.inplace = inplace
        self.train = train
        self.optimize = optimize
        self.dynamic = dynamic
        self.simplify = simplify
        self.opset_version = opset_version

    def export_torchscript(self, model, img, file, optimize):
        # TorchScript model export
        prefix = colorstr('TorchScript:')
        try:
            print(f'\n{prefix} starting export with torch {torch.__version__}...')
            f = file.with_suffix('.torchscript.pt')
            ts = torch.jit.trace(model, img, strict=False)
            (optimize_for_mobile(ts) if optimize else ts).save(f)
            print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
            return ts
        except Exception as e:
            print(f'{prefix} export failure: {e}')

    def export_onnx(self, model, img, file, opset_version, train, dynamic, simplify):
        # ONNX model export
        prefix = colorstr('ONNX:')
        try:
            check_requirements(('onnx', 'onnx-simplifier'))
            import onnx

            print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
            f = file.with_suffix('.onnx')
            torch.onnx.export(model, img, f, verbose=False, opset_version=opset_version,
                              training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                              do_constant_folding=not train,
                              input_names=['images'],
                              output_names=['output'],
                              dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                            'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                            } if dynamic else None)

            # Checks
            model_onnx = onnx.load(f)  # load onnx model
            onnx.checker.check_model(model_onnx)  # check onnx model
            # print(onnx.helper.printable_graph(model_onnx.graph))  # print

            # Simplify
            if simplify:
                try:
                    import onnxsim

                    print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                    model_onnx, check = onnxsim.simplify(
                        model_onnx,
                        dynamic_input_shape=dynamic,
                        input_shapes={'images': list(img.shape)} if dynamic else None)
                    assert check, 'assert check failed'
                    onnx.save(model_onnx, f)
                except Exception as e:
                    print(f'{prefix} simplifier failure: {e}')
            print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        except Exception as e:
            print(f'{prefix} export failure: {e}')

    def export_coreml(self, model, img, file):
        # CoreML model export
        prefix = colorstr('CoreML:')
        try:
            import coremltools as ct

            print(f'\n{prefix} starting export with coremltools {ct.__version__}...')
            f = file.with_suffix('.mlmodel')
            model.train()  # CoreML exports should be placed in model.train() mode
            ts = torch.jit.trace(model, img, strict=False)  # TorchScript model
            model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
            model.save(f)
            print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        except Exception as e:
            print(f'{prefix} export failure: {e}')

    def run(self):
        t = time.time()
        include = [x.lower() for x in self.include]
        img_size = self.img_size * 2 if len(self.img_size) == 1 else 1  # expand
        file = Path(self.weights)

        # Load PyTorch model
        device = select_device(self.device)
        assert not (device.type == 'cpu' and self.half), '--half only compatible with GPU export, i.e. use --device 0'
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        names = model.names

        # Input
        gs = int(max(model.stride))  # grid size (max stride)
        img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
        img = torch.zeros(self.batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection

        # Update model
        if self.half:
            img, model = img.half(), model.half()  # to FP16
        model.train() if self.train else model.eval()  # training mode = no Detect() layer grid construction
        for k, m in model.named_modules():
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            elif isinstance(m, Detect):
                m.inplace = self.inplace
                m.onnx_dynamic = self.dynamic
                # m.forward = m.forward_export  # assign forward (optional)

        for _ in range(2):
            y = model(img)  # dry runs
        print(f"\n{colorstr('PyTorch:')} starting from {self.weights} ({file_size(self.weights):.1f} MB)")

        # Exports
        if 'torchscript' in include:
            self.export_torchscript(model, img, file, self.optimize)
        if 'onnx' in include:
            self.export_onnx(model, img, file, self.opset_version, self.train, self.dynamic, self.simplify)
        if 'coreml' in include:
            self.export_coreml(model, img, file)

        # Finish
        print(f'\nExport complete ({time.time() - t:.2f}s). Visualize with https://github.com/lutzroeder/netron.')



