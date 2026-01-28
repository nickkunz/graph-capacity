## base neural network
class TinyNet(nn.Module):
    """ Simple two-layer MLP with 8 hidden units. """
    def __init__(self, p):
        super().__init__()
        self.fc1 = nn.Linear(p, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

## sklearn wrapper for TinyNet
class TinyNetRegressor(BaseEstimator, RegressorMixin):
    """ Scikit-learn compatible wrapper of TinyNet. """
    def __init__(self, input_dim, quantile = 0.5, lr = 0.01, weight_decay = 0.01, epochs = 2000, device = None):
        self.input_dim = input_dim
        self.quantile = quantile
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device
    def fit(self, X, y):
        X_arr = np.asarray(X, dtype = np.float32)
        y_arr = np.asarray(y, dtype = np.float32).reshape(-1, 1)
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = TinyNet(self.input_dim).to(self.device_)
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr = self.lr,
            weight_decay = self.weight_decay,
        )
        X_tensor = torch.from_numpy(X_arr).to(self.device_)
        y_tensor = torch.from_numpy(y_arr).to(self.device_)
        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model_(X_tensor)
            errors = y_tensor - preds
            loss = torch.max((self.quantile - 1) * errors, self.quantile * errors).mean()
            loss.backward()
            optimizer.step()
        return self
    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("TinyNetRegressor must be fit before calling predict().")
        X_arr = np.asarray(X, dtype = np.float32)
        X_tensor = torch.from_numpy(X_arr).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_tensor).cpu().numpy().ravel()
        return preds


## neural network regressors wrapper
class TinyNetRegressors(BaseEstimator):
    def __init__(self, input_dim, quantile_c = 0.99, quantile_r = 0.5, **kwargs):
        self.input_dim = input_dim
        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.kwargs = kwargs
        self.estimator_c = TinyNetRegressor(
            input_dim = input_dim,
            quantile = quantile_c,
            **kwargs
        )
        self.estimator_r = TinyNetRegressor(
            input_dim = input_dim,
            quantile = quantile_r,
            **kwargs
        )
