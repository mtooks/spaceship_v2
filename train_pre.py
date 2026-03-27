import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import sys

from scipy.ndimage import maximum_filter  # <-- (1) import maximum_filter
from helpers import make_data, score_iou

# Using the same model class from before but with 6 outputs
class SpaceshipDetector6(nn.Module):
    def __init__(self, image_size=200, base_filters=8):
        super().__init__()
        # We match the progression: [1, 2, 4, 8, 16, 32, 64],
        # multiplied by base_filters (8).
        filters_list = [base_filters * i for i in [1, 2, 4, 8, 16, 32, 64]]

        layers = []
        in_channels = 1  # single-channel input

        for out_channels in filters_list:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                    padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        self.conv_stack = nn.Sequential(*layers)
        
        # After 7 pooling operations on a 200x200 image, the spatial size
        # is approximately 200 / (2^7) = 1.5625, which floors to 1 in 
        # typical PyTorch MaxPool settings. So final feature map is [512, 1, 1].
        # => 512 = base_filters*(64).
        
        # The final number of channels is filters_list[-1].
        # Flatten and predict 6 params (x, y, sin_yaw, cos_yaw, w, h).
        self.fc = nn.Linear(filters_list[-1], 6)

    def forward(self, x):
        # x shape: (batch, 1, 200, 200)
        feats = self.conv_stack(x)
        # feats shape likely: (batch, 512, 1, 1)
        feats = feats.view(feats.size(0), -1)  # flatten
        out = self.fc(feats)  # (batch, 6)
        return out
    
class OrientationLoss(nn.Module):
    """
    Expects:
      y_pred: (batch, 6) -> [x, y, sin_yaw, cos_yaw, width, height]
      y_true: (batch, 6) -> same format
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # position (x, y)
        pos_loss = self.mse(y_pred[:, 0:2], y_true[:, 0:2])
        
        # orientation (sin_yaw, cos_yaw)
        ori_loss = self.mse(y_pred[:, 2:4], y_true[:, 2:4])
        
        # size (width, height)
        size_loss = self.mse(y_pred[:, 4:6], y_true[:, 4:6])
        
        total_loss = pos_loss + ori_loss + size_loss
        return total_loss
    
def make_batch(batch_size):
    imgs, labels = [], []
    for _ in range(batch_size):
        img, label = make_data(has_spaceship=True)  # label is [x, y, yaw, w, h]

        # (2) apply maximum_filter preprocessing
        # You can choose the 'footprint' or 'size' parameter as desired:
        # e.g. size=3 or size=(5,5) or a custom footprint.
        # We'll use size=3 as a simple example.
        img_filtered = maximum_filter(img, size=3)
        
        # convert the label => sin/cos
        if not np.any(np.isnan(label)):
            x, y, yaw, w, h = label
            sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)
            label_sin_cos = np.array([x, y, sin_yaw, cos_yaw, w, h])
        else:
            # No spaceship case: keep label as NaNs
            label_sin_cos = np.full(6, np.nan)

        imgs.append(img_filtered)
        labels.append(label_sin_cos)
    
    imgs = np.stack(imgs)     # shape (batch, 200, 200)
    labels = np.stack(labels) # shape (batch, 6)

    # Convert to tensors
    imgs_t   = torch.from_numpy(imgs).float().unsqueeze(1) # (batch, 1, H, W)
    labels_t = torch.from_numpy(labels).float()            # (batch, 6)
    return imgs_t, labels_t

def convert_pred_sin_cos_to_xywhr(pred_params):
    """
    pred_params: [x, y, sin_yaw, cos_yaw, w, h]
    returns: [x, y, yaw, w, h]
    """
    x, y, sin_yaw, cos_yaw, w, h = pred_params
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return np.array([x, y, yaw, w, h])

def get_lr_scheduler(optimizer, config):
    if config.lr_schedule == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.lr_decay_epochs, 
            gamma=config.lr_decay_factor
        )
    elif config.lr_schedule == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs
        )
    return None

def train_model(config=None):
    with wandb.init(project='spaceship-detection', tags = ['lr_scheduler=step'], config=config) as run:
        config = wandb.config
        
        # Model initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SpaceshipDetector6(image_size=200, base_filters=config.base_filters)
        model.to(device)
        
        # Optimizer and scheduler setup
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = get_lr_scheduler(optimizer, config)
        loss_fn = OrientationLoss()
        
        wandb.watch(model)
        
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            
            # --------------------------
            #       Training Loop
            # --------------------------
            for step in range(config.steps_per_epoch):
                imgs, labels = make_batch(config.batch_size)
                imgs, labels = imgs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                preds = model(imgs)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = config.learning_rate
                
            # --------------------------
            #       Validation Loop
            # --------------------------
            model.eval()
            iou_scores = []
            with torch.no_grad():
                for i in range(config.val_samples):
                    img, label = make_data(has_spaceship=True)
                    
                    # Apply the same filter to the validation image
                    img_filtered = maximum_filter(img, size=3)
                    
                    if not np.any(np.isnan(label)):
                        x, y, yaw, w, h = label
                        label = np.array([x, y, np.sin(yaw), np.cos(yaw), w, h])
                    # shape => (1, 1, 200, 200)
                    img_t = torch.from_numpy(img_filtered).float().unsqueeze(0).unsqueeze(0).to(device)
                    label_t = torch.from_numpy(label).float().unsqueeze(0).to(device)

                    pred_6 = model(img_t)  # (1,6)
                    pred_6 = pred_6.squeeze(0).cpu().numpy()  # [x, y, sin_yaw, cos_yaw, w, h]

                    # convert predicted [x, y, sin, cos, w, h] => [x, y, yaw, w, h]
                    pred_5 = convert_pred_sin_cos_to_xywhr(pred_6)
                    
                    # same for label
                    label_5 = convert_pred_sin_cos_to_xywhr(label)

                    iou_val = score_iou(pred_5, label_5)
                    if iou_val is not None:
                        iou_scores.append(iou_val)
            
            mean_iou = np.mean(iou_scores) if len(iou_scores) > 0 else float('nan')
            
            # Log metrics
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": running_loss / config.steps_per_epoch,
                "val_iou": mean_iou,
                "learning_rate": current_lr
            })
            
            print(f"[Epoch {epoch+1}/{config.epochs}] "
                  f"LR: {current_lr:.6f} | "
                  f"Loss: {running_loss/config.steps_per_epoch:.4f} | "
                  f"IoU: {mean_iou:.4f}")
        
        # Save model with run ID in filename
        model_path = f"model_yaw_{run.id}.pt"
        # torch.save(model.state_dict(), model_path)
        # wandb.save(model_path)

def main():
    default_config = {
        "epochs": 30,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "base_filters": 16,
        "steps_per_epoch": 500,
        "val_samples": 100,
        "lr_schedule": "step",
        "lr_decay_epochs": 15,
        "lr_decay_factor": 0.1
    }
    
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        wandb.agent(sys.argv[2], train_model)
    else:
        train_model(default_config)

if __name__ == "__main__":
    main()
