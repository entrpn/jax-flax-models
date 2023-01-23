from vision_transformer import ViT
import jax
import optax
import os
from flax.training import train_state, checkpoints
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import defaultdict
import numpy as np

CHECKPOINT_PATH = "./saved_models/tutorial15_jax"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
class TrainerModule:
    def __init__(self, exmp_imgs, lr=1e-3, weight_decay=0.01, seed=42, **model_hparams):
        super().__init__()
        self.lr = lr
        self.weight_decay=weight_decay
        self.seed=seed
        self.rng=jax.random.PRNGKey(self.seed)
        self.model = ViT(**model_hparams)
        self.log_dir = os.path.join(CHECKPOINT_PATH, 'ViT/')
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.create_functions()
        self.init_model(exmp_imgs)
    
    def create_functions(self):
        def calculate_loss(params, rng,batch,train):
            imgs, labels = batch
            rng, dropout_apply_rng = jax.random.split(rng)
            logits = self.model.apply({
                'params': params},
                imgs,
                train=train,
                rngs={'dropout' : dropout_apply_rng}
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (acc, rng)
        
        def train_step(state, rng, batch):
            loss_fn = lambda params: calculate_loss(params, rng, batch, train=True)
            (loss, (acc,rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, rng, loss, acc
        
        def eval_step(state,rng,batch):
            _, (acc, rng) = calculate_loss(state.params, rng, batch, train=False)
            return rng, acc
        
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)
    
    def init_model(self, exmp_imgs):
        self.rng, init_rng, dropout_init_rng = jax.random.split(self.rng, 3)
        self.init_params = self.model.init({'params' : init_rng, 'dropout' : dropout_init_rng}, exmp_imgs,train=True)['params']
        self.state = None
    
    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.lr,
            boundaries_and_scales={
                int(num_steps_per_epoch*num_epochs*0.6) : 0.1,
                int(num_steps_per_epoch*num_epochs*0.85): 0.1
            }
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr_schedule, weight_decay=self.weight_decay)
        )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            tx=optimizer
        )
    
    def train_model(self, train_loader, val_loader, num_epochs=200):
        self.init_optimizer(num_epochs, len(train_loader))
        best_eval=0.0
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            self.train_epoch(epoch=epoch_idx, train_loader=train_loader)
            if epoch_idx % 10 == 0:
                eval_acc = self.eval_model(val_loader)
                print('eval_acc:',eval_acc)
                self.logger.add_scalar('val/acc',eval_acc, global_step=epoch_idx)
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()
    
    def train_epoch(self, epoch, train_loader):
        metrics = defaultdict(list)
        for batch in tqdm(train_loader, desc='Training', leave=False):
            self.state, self.rng, loss, acc = self.train_step(self.state, self.rng, batch)
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)
        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger.add_scalar('train/'+key, avg_val, global_step=epoch)
    
    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        correct_class, count = 0, 0
        for batch in data_loader:
            self.rng, acc = self.eval_step(self.state, self.rng, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target=self.state.params,
                                    step=step,
                                    overwrite=True)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'ViT.ckpt'), target=None)
        self.state = train_state.TrainState.create(
                                       apply_fn=self.model.apply,
                                       params=params,
                                       tx=self.state.tx if self.state else optax.adamw(self.lr)  # Default optimizer
                                      )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'ViT.ckpt'))