import mxnet as mx


class Model(object):
    def __init__(self):
        self.device = mx.cpu()
        sym, arg_params, aux_params = mx.model.load_checkpoint('DB/feature_extractor/CGD.CUB200.C_concat_MG.ResNet50v.dim1536', 0)
        self._data_shape = (1, 3, 224, 224)
        self._executor = sym.simple_bind(ctx=self.device, data=self._data_shape, grad_req='null', force_rebind=True)
        self._executor.copy_params_from(arg_params, aux_params)

    def __call__(self, data):
        y = self._executor.forward(is_train=False, data=data.as_in_context(self.device))
        embeds = y[0]
        return embeds


class input_feature_map(object):
    def __init__(self, transform, model, device):
        self.transform = transform
        self.model = model
        self.ctx = device

    def get_feature_map(self, image):
        if image is None:
            return None
        image = mx.ndarray.array(image)
        image = self.transform(image)
        image = image.reshape(1, 3, 224, 224)
        features = self.model(image.as_in_context(self.ctx))
        features = features.asnumpy()

        return features
