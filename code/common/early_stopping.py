# Author: Simon Hessner (shessner@cs.cmu.edu)


class EarlyStopping(object):
    def __init__(self, patience=20, max_ratio=0.975, verbose=False):
        """
        Watches validation losses and returns True if a plateau can't be left for a given amount of epochs

        :param patience: If the validation loss is not decreasing (considering the threshold) for patience epochs,
            the EarlyStopping criterion returns True
        :param max_ratio: the ratio new_validation_loss / last_loss must be <= max_ratio in order to reset the
            "failure counter" to 0. Every time the new_validation_loss / last_loss > max_ratio, the counter is increased.
            When the counter exceeds patience, the early stopping criterion is met and training should be stopped.
            The lower max_ratio is set, the more improvement must be achieved by the model in each epoch.
            A very high max_ratio makes it very unlikely that the early stopping criterion is met.
            If max_ratio is set to 1, the validation loss is allowed to stay the same, but increasing is punished
            A max_ratio of 0.95 means that if the model only improves by less than 5% for :patience epochs in a row,
            training will be early stopped
        """
        self.patience = patience
        assert self.patience > 1, "patience must be > 1"

        self.max_ratio = max_ratio
        assert self.max_ratio > 0, "max_ratio must be > 0"
        assert self.max_ratio <= 1, "max_ratio must be <= 1"

        self.last_loss = None
        self.counter = 0
        self.stop = False

        self.verbose = verbose
        if self.verbose:
            print("Early stopping with patience=%d and max_ratio=%f" % (self.patience, self.max_ratio))

        self.enabled = True # if False, the early stopping criterion will always return False

    def info(self, text):
        if self.verbose:
            print("[Early Stopping]", text)

    def __call__(self, new_validation_loss):
        if not self.enabled:
            return False

        if self.stop:
            return True

        ratio = 1 if self.last_loss is None else new_validation_loss / self.last_loss
        tmp_last_loss = self.last_loss if self.last_loss is not None else new_validation_loss
        self.last_loss = new_validation_loss
        if self.last_loss == 0.0:
            self.last_loss = 1e-12 # Avoid division by zero

        if ratio <= self.max_ratio:
            self.counter = 0
            self.info("%f / %f = %f <= %f, counter: %d" % (new_validation_loss, tmp_last_loss, ratio, self.max_ratio, self.counter))
        else:
            self.counter += 1
            self.info("%f / %f = %f > %f, counter: %d" % (new_validation_loss, tmp_last_loss, ratio, self.max_ratio, self.counter))

        if self.counter > self.patience:
            if self.verbose:
                print("Early stopping kicks in")
            self.stop = True
            return True

        return False

    def reset(self):
        self.stop = False
        self.counter = 0
