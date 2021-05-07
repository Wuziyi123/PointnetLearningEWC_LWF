class EarlyStopper:

    def __init__(self, patience, delta=0):
        self._patience = patience
        self._patience_step = 0
        self._delta = delta
        self._best_loss = None
        self._early_stopping = False

    def stop(self, loss):
        if not self._best_loss:
            self._best_loss = loss
        elif loss <= self._best_loss + self._delta:
            self._patience_step += 1
            if self._patience_step >= self._patience:
                self._early_stopping = True
        else:
            self._best_loss = loss
            self._patience_step = 0

    def __bool__(self):
        return self._early_stopping
