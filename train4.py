import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from torch.nn import functional as F

from helpers import make_data, score_iou

# Using the same model class from before but with 6 outputs
class SpaceshipDetector6(nn.Module):
    def __init__(self, image_size=200, base_filters=8):
        super().__init__()
        # We match the progression: [1, 2, 4, 8, 16, 32, 64],
        # multiplied by base_filters (8).
        filters_list = [base_filters * i for i in [1, 32, 32, 64, 128, 128, 128,256,256,256]]

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
        #  => 512 = base_filters*(64).
        
        # The final number of channels is filters_list[-1].
        # Flatten and predict 5 params (x, y, yaw, width, height).
        self.fc1 = nn.Linear(filters_list[-1], 64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Linear(64, 6)
    def forward(self, x):
        # x shape: (batch, 1, 200, 200)
        feats = self.conv_stack(x)
        feats = self.avgpool(feats)
        # feats shape likely: (batch, 512, 1, 1)
        feats = feats.view(feats.size(0), -1)  # flatten
        feats = self.fc1(feats)
        feats = self.fc2(feats)
        return feats # remember to output 6
    
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
        # Ensure inputs are 2D tensors
        # print(type(y_pred))
        y_pred = y_pred.view(-1, 6)
        y_true = y_true.view(-1, 6)
        
        # position (x, y)
        pos_loss = F.smooth_l1_loss(y_pred[:, 0:2], y_true[:, 0:2])
        size_loss = F.smooth_l1_loss(y_pred[:, 4:6], y_true[:, 4:6])
        
        # orientation (sin_yaw, cos_yaw)
        ori_loss = self.mse(y_pred[:, 2:4], y_true[:, 2:4])
        
        # size (width, height)
        
        
        total_loss = pos_loss + ori_loss + size_loss
        return total_loss
    
def make_batch(batch_size):
    imgs, labels = [], []
    for _ in range(batch_size):
        img, label = make_data(has_spaceship=True)  # label is [x, y, yaw, w, h]
        if not np.any(np.isnan(label)):
            x, y, yaw, w, h = label
            sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)
            label_sin_cos = np.array([x, y, sin_yaw, cos_yaw, w, h])
        else:
            # No spaceship case: keep label as NaNs or
            # you could define a "no-ship" format. 
            label_sin_cos = np.full(6, np.nan)

        imgs.append(img)
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

def main():
    # Initialize wandb
    # wandb.init(mode="disabled")
    wandb.init(
        # onlien or disabled
        mode='disabled',
        project="spaceship-detection",
        name="yolo",
        tags=['mse_loss'],
        config={
            "epochs": 30,
            # batch size should be 64 
            "batch_size": 32,
            "learning_rate": 1e-3,
            "base_filters": 1,
            "steps_per_epoch": 500,
            "val_samples": 100
        }
    )
    
    NUM_EPOCHS = wandb.config.epochs
    BATCH_SIZE = wandb.config.batch_size
    STEPS_PER_EPOCH = wandb.config.steps_per_epoch
    VAL_SAMPLES = wandb.config.val_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #IMPORTANT: base_filters=4 breaks everything!

    model = SpaceshipDetector6(image_size=200, base_filters=wandb.config.base_filters)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    model.to(device)
    
    # Log model graph and gradients
    wandb.watch(model)

    loss_fn = OrientationLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)


    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for step in range(STEPS_PER_EPOCH):
            imgs, labels = make_batch(BATCH_SIZE)  # now returns (batch, 6) labels
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            
            # Compute your orientation-aware loss
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / STEPS_PER_EPOCH
        
        # Validation code
        model.eval()
        iou_scores = []
        validation_losses = []
        
        with torch.no_grad():
            for i in range(VAL_SAMPLES):
                img, label = make_data(has_spaceship=True)
                # Convert label => sin/cos format if needed
                if not np.any(np.isnan(label)):
                    x, y, yaw, w, h = label
                    label = np.array([x, y, np.sin(yaw), np.cos(yaw), w, h])
                # shape => (1, 1, 200, 200)
                img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
                label_t = torch.from_numpy(label).float().unsqueeze(0).to(device)

                pred_6 = model(img_t)  # (1,6)

                validation_losses.append(loss_fn(pred_6, label_t).item())
                pred_6 = pred_6.squeeze(0).cpu().numpy()  # [x, y, sin_yaw, cos_yaw, w, h]

                # pred_6 = pred_6.squeeze(0).cpu().numpy() 
                #REPLACE # [x, y, sin_yaw, cos_yaw, w, h]

                # convert predicted [x, y, sin, cos, w, h] => [x, y, yaw, w, h]
                pred_5 = convert_pred_sin_cos_to_xywhr(pred_6)
                
                # same for label
                # print(i)
                label_5 = convert_pred_sin_cos_to_xywhr(label)

                iou_val = score_iou(pred_5, label_5)
                
                if iou_val is not None:
                    iou_scores.append(iou_val)

        mean_iou = np.mean(iou_scores) if len(iou_scores) > 0 else float('nan')
        mean_val_loss = np.mean(validation_losses)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": mean_val_loss,
            "val_iou": mean_iou,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | Val IoU: {mean_iou:.4f}")
        
    # Save model
    # model_artifact = wandb.Artifact(
    #     "spaceship_detector", type="model",
    #     description="Spaceship detection model with yaw prediction"
    # )
    torch.save(model.state_dict(), "model_yolo.pt")
    # model_artifact.add_file("model_yaw.pt")
    # wandb.log_artifact(model_artifact)
    
    wandb.finish()
    print("Training complete. Model saved to model_yaw.pt")

if __name__ == "__main__":
    main()