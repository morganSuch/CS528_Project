# %%
# %%
import torch
import pytorch_lightning as pl
import torchvision as tv
import os

from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from sklearn import metrics

from PIL import Image
from collections import Counter
from tqdm import tqdm

from deepee import UniformDataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class Arguments:
    def __init__(self):
        self.batch_size = 32
        self.test_batch_size = 700
        self.lr = 1e-4
        self.num_epochs = 20
        self.log_interval = 100
        self.optimizer = 'sgd'

args = Arguments()

def single_channel_loader(filename):
    """Converts `filename` to a grayscale PIL Image"""
    with open(filename, "rb") as f:
        img = Image.open(f).convert("L")
        return img.copy()

# takes in the Image folder, does some transformations, and loads the image
# with single_channel_loader
trainset = tv.datasets.ImageFolder(
    "/home/nikseres/pneu/Pediatric Chest X-ray Pneumonia/train",
    transform=tv.transforms.Compose(
        [
            tv.transforms.RandomAffine(
                degrees=45, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            tv.transforms.Resize(224),
            tv.transforms.RandomCrop((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4814,), (0.2377,)),
            tv.transforms.Lambda(
                lambda x: torch.repeat_interleave(  # pylint: disable=no-member
                    x, 3, dim=0
                )
            ),
        ]
    ),
    # target_transform=tv.transforms.ToTensor(),
    loader=single_channel_loader,
)

# randomly split the train and validation sets
L_train = round(0.85 * len(trainset))
trainset, valset = torch.utils.data.random_split(
    trainset,
    (L_train, len(trainset) - L_train),
    generator=torch.Generator().manual_seed(42),
)

# load in the testing set the same way
testset = tv.datasets.ImageFolder(
    "/home/nikseres/pneu/Pediatric Chest X-ray Pneumonia/test",
    transform=tv.transforms.Compose(
        [
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4814,), (0.2377,)),
            tv.transforms.Lambda(
                lambda x: torch.repeat_interleave(  # pylint: disable=no-member
                    x, 3, dim=0
                )
            ),
        ]
    ),
    # target_transform=tv.transforms.ToTensor(),
    loader=single_channel_loader,
)

trainloader = UniformDataLoader(
    trainset,
    batch_size=args.batch_size,
    pin_memory=torch.cuda.is_available(),
    num_workers=1 if torch.cuda.is_available() else 4,
)
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batch_size,
    pin_memory=torch.cuda.is_available(),
    num_workers=1 if torch.cuda.is_available() else 4,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.test_batch_size,
    pin_memory=torch.cuda.is_available(),
    num_workers=1 if torch.cuda.is_available() else 4,
    shuffle=True,
)

img_batch = torch.stack([trainset[i][0] for i in range(32)])
grid_img = make_grid(img_batch, nrow=8)
grid_img = (grid_img - torch.min(grid_img)) / (
    torch.max(grid_img) - torch.min(grid_img)
)
# save a random set of images from the training set 
plt.figure(figsize=(10, 5))
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.savefig("grid.png")

# t1dm is a smart loading bar 
targets = []
for _, target in tqdm(trainloader, total=len(trainloader), leave=False):
    targets.extend(target.tolist())
# %%
# set some class weights according to the target distributions
target_distribution = Counter(targets)
class_weights = torch.tensor(
    [
        1.0 - (target_distribution[key] / len(trainset))
        for key in sorted(target_distribution)
    ]
) * len(target_distribution)

# %%
class BaseClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.save_hyperparameters()

    def forward(self, x):
        embedding = self.classifier(x)
        return embedding

    def training_step(self, batch, batch_idx):
        data, target = batch
        pred = self.classifier(data).squeeze()
        loss = self.loss_fn(
            pred,
            target.to(torch.float),
            weight=self.weight[target.data.view(-1).long()]
            .view_as(target)
            .to(self.device),
        )
        self.log("train_loss", loss)
        output = {"loss": loss}
        return output

    def configure_optimizers(self):
        return torch.optim(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        data, target = batch
        pred = self.classifier(data).squeeze()
        loss = self.loss_fn(pred, target.to(torch.float))
        pred = torch.sigmoid(pred.detach().cpu())
        pred_classes = torch.where(pred < 0.5, 0, 1).tolist()
        target = target.cpu().detach().tolist()
        mcc = metrics.matthews_corrcoef(target, pred_classes)
        acc = metrics.accuracy_score(target, pred_classes)
        roc = metrics.roc_auc_score(target, pred)
        out = {"val_loss": loss, "val_mcc": mcc, "val_accuracy": acc, "val_roc": roc}
        for key, value in out.items():
            self.log(key, value)
        return out

    def test_step(self, batch, batch_idx):
        data, target = batch
        target = target
        pred = self.classifier(data).squeeze()
        loss = self.loss_fn(pred, target.to(torch.float))
        pred = torch.sigmoid(pred.detach().cpu())
        pred_classes = torch.where(pred < 0.5, 0, 1).tolist()
        target = target.cpu().detach().tolist()
        mcc = metrics.matthews_corrcoef(target, pred_classes)
        acc = metrics.accuracy_score(target, pred_classes)
        roc = metrics.roc_auc_score(target, pred)
        out = {
            "test_loss": loss,
            "test_mcc": mcc,
            "test_accuracy": acc,
            "test_roc": roc,
        }
        for key, value in out.items():
            self.log(key, value)
        return out

    def on_epoch_start(self):
        if self.current_epoch == 1:
            print("Unfreeze normalization layers")
            for layer_name, layer in self.classifier.named_modules():
                if (
                    isinstance(layer, torch.nn.BatchNorm2d)
                    or isinstance(layer, torch.nn.GroupNorm)
                    or isinstance(layer, torch.nn.Linear)
                ):
                    for p in layer.parameters():
                        p.requires_grad = True
        if self.current_epoch == 2:
            print("Unfreeze model")
            for param in self.classifier.parameters():
                param.requires_grad = True

    def test(self, testloader):
        self.classifier.eval()
        preds, targets = [], []
        total_loss = 0.0
        with torch.no_grad():
            for data, target in testloader:
                pred = self.classifier(data.to(self.device)).squeeze()
                loss = self.loss_fn(pred, target.to(torch.float))
                preds.append(torch.sigmoid(pred.detach().cpu()))
                targets.extend(target.detach().cpu().tolist())
                total_loss += loss.detach().cpu().item()

        preds = torch.vstack(preds).squeeze()
        pred_classes = torch.where(preds < 0.5, 0, 1).tolist()
        mcc = metrics.matthews_corrcoef(targets, pred_classes)
        acc = metrics.accuracy_score(targets, pred_classes)
        roc = metrics.roc_auc_score(targets, pred)
        out = {
            "test_loss": total_loss / len(testloader),
            "test_mcc": mcc,
            "test_accuracy": acc,
            "test_roc": roc,
        }
        return out

# %%
from deepee import ModelSurgeon, SurgicalProcedures

surgeon = ModelSurgeon(SurgicalProcedures.BN_to_BN_nostats)
logger = pl.loggers.TensorBoardLogger("logs", name="pneumonia_classification_secondrun_baseline")

# %%
class PLClassifier(BaseClassifier):
    def __init__(self, args, class_weights, transfer_learning=True):
        super().__init__(args)
        self.classifier = tv.models.vgg11_bn(pretrained=True)
        self.classifier.classifier = torch.nn.Linear(512 * 7 * 7, 1)
        if transfer_learning:
            for param in self.classifier.parameters():
                param.requires_grad = False
            for param in self.classifier.classifier.parameters():
                param.requires_grad = True
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        self.weight = class_weights
        surgeon.operate(self.classifier)  # for comparison reasons

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

classifier = PLClassifier(args, class_weights, transfer_learning=False)

# args.lr = 5e-4
# # %%
# results = []
# # for _ in range(10):
# classifier = PLClassifier(args, class_weights)
# trainer = pl.Trainer(
#     max_epochs=args.num_epochs,
#     gpus=1 if torch.cuda.is_available() else 0,
#     logger=logger
#     # overfit_batches=2
# )
# trainer.fit(classifier, trainloader, valloader)

# res = classifier.test(testloader)
# print(res)
# results.append(res)
# # %%
# torch.save(results, "vanilla_results_secondrun.pt")


from deepee.watchdog import PrivacyWatchdog, PrivacyBudgetExhausted
from deepee import PrivacyWrapper

# %%
# %%
# now privately
noise_mult = 0.5
clip_norm = 1.0

class PrivatePLClassifier(BaseClassifier):
    def __init__(self, args, class_weights, transfer_learning=True):
        super().__init__(args)
        self.args = args
        model = tv.models.vgg11_bn(pretrained=True)
        model.classifier = torch.nn.Linear(512 * 7 * 7, 1)
        if transfer_learning:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        self.weight = class_weights
        surgeon.operate(model)
        watchdog = PrivacyWatchdog(
            trainloader,
            target_epsilon=100.0,
            abort=True,
            target_delta=1e-5,
            report_every_n_steps=len(trainloader),
            # fallback_to_rdp=True,
        )
        self.classifier = PrivacyWrapper(
            model,
            args.batch_size,
            L2_clip=clip_norm,
            noise_multiplier=noise_mult,
            watchdog=watchdog,  # watchdog,
        )

    def on_after_backward(self):
        self.classifier.clip_and_accumulate()
        self.classifier.noise_gradient()

    def optimizer_step(self, *arg, **kwargs):
        super().optimizer_step(*arg, **kwargs)
        self.classifier.prepare_next_batch()

    def on_epoch_start(self):
        if self.current_epoch == 1:
            print("Unfreeze normalization layers")
            for layer_name, layer in self.classifier.wrapped_model.named_modules():
                if (
                    isinstance(layer, torch.nn.BatchNorm2d)
                    or isinstance(layer, torch.nn.GroupNorm)
                    or isinstance(layer, torch.nn.Linear)
                ):
                    for p in layer.parameters():
                        p.requires_grad = True
            self.classifier.update_clones()
        if self.current_epoch == 2:
            print("Unfreeze model")
            for param in self.classifier.wrapped_model.parameters():
                param.requires_grad = True
            self.classifier.update_clones()

    def configure_optimizers(self):
        if args.optimizer == 'sgd': 
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

# %%
# SGD run 
args.optimizer = 'sgd'
privateclassifier = PrivatePLClassifier(args, class_weights, transfer_learning=False)
temp_tr = pl.Trainer()

# %%
# from lr finder
args.lr = 5e-3
args.num_epochs = 20

# %%
results = []
# for _ in range(10):
print("Current private epsilon: {}".format(privateclassifier.classifier.current_epsilon))
privateclassifier = PrivatePLClassifier(args, class_weights)
logger = pl.loggers.TensorBoardLogger("logs", name="pneumonia_classification_secondrun_sgd_deepee")
privatetrainer = pl.Trainer(
    max_epochs=args.num_epochs,
    gpus=1 if torch.cuda.is_available() else 0,
    logger=logger,
    callbacks=[pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    # overfit_batches=1,
)
try:
    privatetrainer.fit(privateclassifier, trainloader, valloader)
except PrivacyBudgetExhausted as e:
    print(f"Privacy budget is exhausted")

print(f"Final epsilon: {privateclassifier.classifier.current_epsilon:.2f}")
res = privateclassifier.test(testloader)
print(res)
results.append(res)
# %%
torch.save(results, "DP_results_small_epsilon_newrun_sgd_secondrun.pt")
# rdp version
from deepee.watchdog import compute_rdp, rdp_privacy_spent

q = args.batch_size / len(trainset)
orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
steps_taken = args.num_epochs * len(trainloader)
rdp = compute_rdp(q, noise_mult, steps_taken, orders)
spent, _ = rdp_privacy_spent(orders=orders, rdp=rdp, delta=1e-5)
print(f"RDP epsilon: {spent}")

# ------------------------------------------

# %%
# Adam run 
args.optimizer = 'adam'
privateclassifier = PrivatePLClassifier(args, class_weights, transfer_learning=False)
temp_tr = pl.Trainer()

# %%
# from lr finder
args.lr = 5e-3
args.num_epochs = 20

# %%
results = []
# for _ in range(10):
print("Current private epsilon: {}".format(privateclassifier.classifier.current_epsilon))
privateclassifier = PrivatePLClassifier(args, class_weights)
logger = pl.loggers.TensorBoardLogger("logs", name="pneumonia_classification_secondrun_adam")
privatetrainer = pl.Trainer(
    max_epochs=args.num_epochs,
    gpus=1 if torch.cuda.is_available() else 0,
    logger=logger,
    callbacks=[pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    # overfit_batches=1,
)
try:
    privatetrainer.fit(privateclassifier, trainloader, valloader)
except PrivacyBudgetExhausted as e:
    print(f"Privacy budget is exhausted")

print(f"Final epsilon: {privateclassifier.classifier.current_epsilon:.2f}")
res = privateclassifier.test(testloader)
print(res)
results.append(res)
# %%
torch.save(results, "DP_results_small_epsilon_newrun_adam_secondrun.pt")
# rdp version
from deepee.watchdog import compute_rdp, rdp_privacy_spent

q = args.batch_size / len(trainset)
orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
steps_taken = args.num_epochs * len(trainloader)
rdp = compute_rdp(q, noise_mult, steps_taken, orders)
spent, _ = rdp_privacy_spent(orders=orders, rdp=rdp, delta=1e-5)
print(f"RDP epsilon: {spent}")

# ----------------------------

# %%
# Adagrad run 
args.optimizer = 'adagrad'
privateclassifier = PrivatePLClassifier(args, class_weights, transfer_learning=False)
temp_tr = pl.Trainer()

# %%
# from lr finder
args.lr = 5e-3
args.num_epochs = 20

# %%
results = []
# for _ in range(10):
print("Current private epsilon: {}".format(privateclassifier.classifier.current_epsilon))
privateclassifier = PrivatePLClassifier(args, class_weights)
logger = pl.loggers.TensorBoardLogger("logs", name="pneumonia_classification_secondrun_adagrad")
privatetrainer = pl.Trainer(
    max_epochs=args.num_epochs,
    gpus=1 if torch.cuda.is_available() else 0,
    logger=logger,
    callbacks=[pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    # overfit_batches=1,
)
try:
    privatetrainer.fit(privateclassifier, trainloader, valloader)
except PrivacyBudgetExhausted as e:
    print(f"Privacy budget is exhausted")

print(f"Final epsilon: {privateclassifier.classifier.current_epsilon:.2f}")
res = privateclassifier.test(testloader)
print(res)
results.append(res)
# %%
torch.save(results, "DP_results_small_epsilon_secondrun_adagrad.pt")
# rdp version
from deepee.watchdog import compute_rdp, rdp_privacy_spent

q = args.batch_size / len(trainset)
orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
steps_taken = args.num_epochs * len(trainloader)
rdp = compute_rdp(q, noise_mult, steps_taken, orders)
spent, _ = rdp_privacy_spent(orders=orders, rdp=rdp, delta=1e-5)
print(f"RDP epsilon: {spent}")
