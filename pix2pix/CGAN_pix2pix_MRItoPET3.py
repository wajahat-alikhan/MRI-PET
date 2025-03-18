#The best performance one, with validation has better performance than training

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import numpy as np
from pathlib import Path
import re
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Set up device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# UNet building block
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Generator (UNet3D Architecture)
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()
        
        features = init_features
        
        # Encoder path
        self.encoder1 = DoubleConv3D(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        self.encoder2 = DoubleConv3D(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        self.encoder3 = DoubleConv3D(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        # Bridge
        self.bridge = DoubleConv3D(features * 4, features * 8)
        
        # Decoder path
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, 
            kernel_size=2, stride=2
        )
        self.decoder3 = DoubleConv3D(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2,
            kernel_size=2, stride=2
        )
        self.decoder2 = DoubleConv3D(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features,
            kernel_size=2, stride=2
        )
        self.decoder1 = DoubleConv3D(features * 2, features)
        
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        input_size = x.size()
        
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool3(enc3)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder with skip connections and size adjustments
        x = self.upconv3(x)
        enc3 = self._adjust_size(enc3, x)
        x = torch.cat((x, enc3), dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        enc2 = self._adjust_size(enc2, x)
        x = torch.cat((x, enc2), dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        enc1 = self._adjust_size(enc1, x)
        x = torch.cat((x, enc1), dim=1)
        x = self.decoder1(x)
        
        x = self.final_conv(x)
        
        if x.size() != input_size:
            x = self._adjust_size(x, target_size=input_size)
        
        return x

    def _adjust_size(self, x, target_tensor=None, target_size=None):
        if target_tensor is not None:
            target_size = target_tensor.size()
        
        if x.size() == target_size:
            return x
            
        pad_dims = []
        for i, (src, dst) in enumerate(zip(x.shape[2:], target_size[2:])):
            pad_amt = dst - src
            if pad_amt > 0:
                pad_dims.extend([pad_amt//2, pad_amt - pad_amt//2])
            elif pad_amt < 0:
                crop = -pad_amt
                x = x.narrow(i+2, crop//2, src - crop)
                pad_dims.extend([0, 0])
            else:
                pad_dims.extend([0, 0])
                
        if any(p != 0 for p in pad_dims):
            x = F.pad(x, pad_dims)
            
        return x
    
# Discriminator Architecture
class Discriminator3D(nn.Module):
    def __init__(self, input_channels=2, ndf=32):  # Starting with fewer filters (32 instead of 64)
        super(Discriminator3D, self).__init__()
        
        # Initial layer without batch normalization
        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2)  # Added dropout to prevent discriminator from becoming too strong
        )
        
        # Main discriminator layers with increasing channel depth
        self.layer1 = self._make_layer(ndf, ndf * 2)        # 32 -> 64 channels
        self.layer2 = self._make_layer(ndf * 2, ndf * 4)    # 64 -> 128 channels
        self.layer3 = self._make_layer(ndf * 4, ndf * 8)    # 128 -> 256 channels
        
        # Final classification layer
        self.final = nn.Sequential(
            nn.Conv3d(ndf * 8, 1, 4, stride=1, padding=1),
            nn.Dropout3d(0.2)  # Added dropout to final layer
        )
        
    def _make_layer(self, in_channels, out_channels):
        """Creates a discriminator layer with batch norm, LeakyReLU, and dropout"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2)
        )
        
    def forward(self, mri, pet):
        # Concatenate MRI and PET along channel dimension
        x = torch.cat([mri, pet], dim=1)
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.final(x)

# Dataset class for handling MRI-PET pairs
class BrainModalityGANDataset(data.Dataset):
    def __init__(self, data_dir: str, is_train=True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
            
        print(f"Scanning directory: {self.data_dir}")
        self.pairs = self._find_pairs()
        
        if len(self.pairs) == 0:
            raise ValueError("No valid MRI-PET pairs found in the directory")
            
        print(f"Successfully initialized dataset with {len(self.pairs)} pairs")
    
    def _find_pairs(self) -> list:
        """Find matching MRI-PET pairs using ADNI filename pattern"""
        pairs = []
        nii_files = list(self.data_dir.glob("*.nii"))
        print(f"Found {len(nii_files)} total .nii files")
        
        pairs_dict = {}
        for file_path in nii_files:
            pattern = r'(\d+_S_\d+)_(\d{4}[-_]\d{2})'
            match = re.search(pattern, file_path.name)
            if not match:
                print(f"Warning: Could not parse filename pattern for {file_path.name}")
                continue
            
            subject_id, date = match.groups()
            key = f"{subject_id}_{date.replace('-', '_')}"
            
            if 'MPRAGE' in file_path.name:
                if key not in pairs_dict:
                    pairs_dict[key] = {'mri': None, 'pet': None}
                pairs_dict[key]['mri'] = file_path
            elif 'PET_FDG_Coreg_Ave' in file_path.name:
                if key not in pairs_dict:
                    pairs_dict[key] = {'mri': None, 'pet': None}
                pairs_dict[key]['pet'] = file_path
        
        for key, pair_dict in pairs_dict.items():
            if pair_dict['mri'] and pair_dict['pet']:
                pairs.append((pair_dict['mri'], pair_dict['pet']))
                
        return pairs

    def _load_and_preprocess(self, file_path: Path) -> torch.Tensor:
        """Load and preprocess NIfTI files with proper normalization"""
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # Handle NaN values
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=np.nanmedian(data))
            
            # Normalize the data
            data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            tensor = torch.FloatTensor(data).unsqueeze(0)
            
            return tensor
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {str(e)}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mri_path, pet_path = self.pairs[idx]
        mri_tensor = self._load_and_preprocess(mri_path)
        pet_tensor = self._load_and_preprocess(pet_path)
        return mri_tensor, pet_tensor
# Loss and metric calculation functions
def calculate_metrics(pred, target):
    """
    Calculate PSNR and SSIM metrics for the predicted and target PET images.
    We calculate these metrics on middle slices for computational efficiency.
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    psnr_vals = []
    ssim_vals = []
    
    for i in range(pred_np.shape[0]):
        # Calculate metrics for middle slices in each dimension
        z_slice = pred_np.shape[2] // 2
        
        pred_slice = pred_np[i, 0, :, :, z_slice]
        target_slice = target_np[i, 0, :, :, z_slice]
        
        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        psnr_val = psnr(target_slice, pred_slice, 
                       data_range=target_slice.max() - target_slice.min())
        psnr_vals.append(psnr_val)
        
        # Calculate SSIM (Structural Similarity Index)
        ssim_val = ssim(target_slice, pred_slice,
                       data_range=target_slice.max() - target_slice.min())
        ssim_vals.append(ssim_val)
    
    return np.mean(psnr_vals), np.mean(ssim_vals)

def generator_loss(disc_generated_output, gen_output, target, lambda_l1=100):
    """
    Calculate generator loss combining adversarial and L1 components.
    lambda_l1 controls the balance between GAN loss and L1 loss.
    """
    loss_object = nn.BCEWithLogitsLoss()
    # Adversarial loss - generator wants discriminator to think its outputs are real
    gan_loss = loss_object(
        disc_generated_output,
        torch.ones_like(disc_generated_output)
    )
    
    # L1 loss - measures absolute pixel-wise differences
    l1_loss = torch.mean(torch.abs(target - gen_output))
    
    # Combined loss with weighting
    total_gen_loss = gan_loss + (lambda_l1 * l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Calculate discriminator loss.
    The discriminator tries to classify real images as 1 and generated images as 0.
    """
    loss_object = nn.BCEWithLogitsLoss()
    
    # Loss for real images
    real_loss = loss_object(
        disc_real_output,
        torch.ones_like(disc_real_output)
    )
    
    # Loss for generated (fake) images
    generated_loss = loss_object(
        disc_generated_output,
        torch.zeros_like(disc_generated_output)
    )
    
    # Total discriminator loss is the average of real and fake losses
    total_disc_loss = (real_loss + generated_loss) * 0.5
    return total_disc_loss

class GANTrainer:
    """
    Trainer class for 3D Pix2Pix GAN with UNet generator.
    Handles the complete training process including visualization and checkpointing.
    """
    def __init__(self, generator, discriminator, train_loader, val_loader,
                 device, num_epochs=50, save_dir='3d_pix2pix_results_after_long_time_50Epochs'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Create directories for saving results
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize optimizers with different learning rates
        self.g_optimizer = optim.Adam(generator.parameters(), 
                                    lr=1e-4, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(discriminator.parameters(), 
                                    lr=4e-5, betas=(0.5, 0.999))
        
        # TensorBoard writer for logging
        self.writer = SummaryWriter('runs/3d_pix2pix_training')
        
        # Initialize metric tracking
        self.train_metrics = {
            'g_loss': [], 'd_loss': [], 'psnr': [], 'ssim': []
        }
        self.val_metrics = {
            'g_loss': [], 'd_loss': [], 'psnr': [], 'ssim': []
        }
    
    def save_prediction_examples(self, mri, pred_pet, true_pet, epoch):
        """Save example predictions showing multiple views."""
        save_dir = self.save_dir / 'visualizations'
        save_dir.mkdir(exist_ok=True)
        
        # Get middle slices for visualization
        z_slice = mri.shape[2] // 2
        y_slice = mri.shape[3] // 2
        x_slice = mri.shape[4] // 2
        
        for batch_idx in range(min(mri.shape[0], 4)):  # Save up to 4 examples
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            
            # Plot axial view
            axes[0, 0].imshow(mri[batch_idx, 0, :, :, z_slice].cpu(), cmap='gray')
            axes[0, 0].set_title('MRI')
            axes[0, 1].imshow(pred_pet[batch_idx, 0, :, :, z_slice].cpu(), cmap='hot')
            axes[0, 1].set_title('Generated PET')
            axes[0, 2].imshow(true_pet[batch_idx, 0, :, :, z_slice].cpu(), cmap='hot')
            axes[0, 2].set_title('True PET')
            
            # Plot coronal view
            axes[1, 0].imshow(mri[batch_idx, 0, :, y_slice, :].cpu(), cmap='gray')
            axes[1, 1].imshow(pred_pet[batch_idx, 0, :, y_slice, :].cpu(), cmap='hot')
            axes[1, 2].imshow(true_pet[batch_idx, 0, :, y_slice, :].cpu(), cmap='hot')
            
            # Plot sagittal view
            axes[2, 0].imshow(mri[batch_idx, 0, x_slice, :, :].cpu(), cmap='gray')
            axes[2, 1].imshow(pred_pet[batch_idx, 0, x_slice, :, :].cpu(), cmap='hot')
            axes[2, 2].imshow(true_pet[batch_idx, 0, x_slice, :, :].cpu(), cmap='hot')
            
            plt.tight_layout()
            plt.savefig(save_dir / f'epoch_{epoch}_sample_{batch_idx}.png')
            plt.close()

    def train_epoch(self):
        """Train for one epoch with improved GAN balance."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc='Training', leave=False) as pbar:
            for i, (mri, pet) in enumerate(pbar):
                mri, pet = mri.to(self.device), pet.to(self.device)
                
                # Train discriminator less frequently (every 3rd iteration)
                if i % 3 == 0:
                    self.d_optimizer.zero_grad()
                    generated_pet = self.generator(mri)
                    
                    real_output = self.discriminator(mri, pet)
                    fake_output = self.discriminator(mri, generated_pet.detach())
                    
                    d_loss = discriminator_loss(real_output, fake_output)
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                    epoch_d_loss += d_loss.item()
                
                # Train generator
                self.g_optimizer.zero_grad()
                generated_pet = self.generator(mri)
                fake_output = self.discriminator(mri, generated_pet)
                
                g_total_loss, g_gan_loss, g_l1_loss = generator_loss(
                    fake_output, generated_pet, pet)
                
                g_total_loss.backward()
                self.g_optimizer.step()
                
                # Calculate metrics
                psnr_val, ssim_val = calculate_metrics(generated_pet, pet)
                
                # Update running averages
                epoch_g_loss += g_total_loss.item()
                epoch_psnr += psnr_val
                epoch_ssim += ssim_val
                
                pbar.set_postfix({
                    'g_loss': f'{g_total_loss.item():.4f}',
                    'd_loss': f'{d_loss.item():.4f}',
                    'psnr': f'{psnr_val:.2f}',
                    'ssim': f'{ssim_val:.4f}'
                })
        
        # Calculate epoch averages
        metrics = {
            'g_loss': epoch_g_loss / num_batches,
            'd_loss': epoch_d_loss / (num_batches // 3),  # Adjust for less frequent D updates
            'psnr': epoch_psnr / num_batches,
            'ssim': epoch_ssim / num_batches
        }
        
        return metrics
    def validate(self):
        """Run validation with metrics tracking and visualization."""
        self.generator.eval()
        self.discriminator.eval()
        
        val_g_loss = 0
        val_d_loss = 0
        val_psnr = 0
        val_ssim = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for i, (mri, pet) in enumerate(self.val_loader):
                mri, pet = mri.to(self.device), pet.to(self.device)
                
                # Generate fake PET scans
                generated_pet = self.generator(mri)
                
                # Calculate discriminator outputs
                real_output = self.discriminator(mri, pet)
                fake_output = self.discriminator(mri, generated_pet)
                
                # Calculate losses
                d_loss = discriminator_loss(real_output, fake_output)
                g_total_loss, _, _ = generator_loss(fake_output, generated_pet, pet)
                
                # Calculate quality metrics
                psnr_val, ssim_val = calculate_metrics(generated_pet, pet)
                
                val_g_loss += g_total_loss.item()
                val_d_loss += d_loss.item()
                val_psnr += psnr_val
                val_ssim += ssim_val
                
                # Save examples from first batch
                if i == 0:
                    self.save_prediction_examples(mri, generated_pet, pet, len(self.val_metrics['g_loss']))
        
        # Calculate validation averages
        metrics = {
            'g_loss': val_g_loss / num_batches,
            'd_loss': val_d_loss / num_batches,
            'psnr': val_psnr / num_batches,
            'ssim': val_ssim / num_batches
        }
        
        return metrics

    def plot_metrics(self):
        """Plot and save training and validation metrics."""
        metrics_dir = self.save_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        # Plot losses
        plt.figure(figsize=(12, 8))
        plt.plot(self.train_metrics['g_loss'], label='Train Generator Loss')
        plt.plot(self.train_metrics['d_loss'], label='Train Discriminator Loss')
        plt.plot(self.val_metrics['g_loss'], label='Val Generator Loss')
        plt.plot(self.val_metrics['d_loss'], label='Val Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig(metrics_dir / 'losses.png')
        plt.close()
        
        # Plot PSNR and SSIM
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.train_metrics['psnr'], label='Train')
        ax1.plot(self.val_metrics['psnr'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('PSNR over Training')
        ax1.legend()
        
        ax2.plot(self.train_metrics['ssim'], label='Train')
        ax2.plot(self.val_metrics['ssim'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('SSIM')
        ax2.set_title('SSIM over Training')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(metrics_dir / 'quality_metrics.png')
        plt.close()

    def train(self):
        """Main training loop with comprehensive logging and checkpointing."""
        print("Starting training...")
        start_time = time.time()
        best_val_psnr = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_metrics = self.train_epoch()
            for key, value in train_metrics.items():
                self.train_metrics[key].append(value)
            
            # Validation phase
            val_metrics = self.validate()
            for key, value in val_metrics.items():
                self.val_metrics[key].append(value)
            
            # Log metrics to TensorBoard
            for key in train_metrics:
                self.writer.add_scalar(f'train/{key}', train_metrics[key], epoch)
                self.writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
            
            # Save best model based on PSNR
            if val_metrics['psnr'] > best_val_psnr:
                best_val_psnr = val_metrics['psnr']
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                    'best_psnr': best_val_psnr,
                }, self.save_dir / 'best_model.pt')
                print(f"\nSaved new best model with PSNR: {best_val_psnr:.2f}")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                    'train_metrics': self.train_metrics,
                    'val_metrics': self.val_metrics,
                }, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
            
            # Plot metrics
            self.plot_metrics()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("Training Metrics:")
            print(f"G_loss: {train_metrics['g_loss']:.4f}, D_loss: {train_metrics['d_loss']:.4f}")
            print(f"PSNR: {train_metrics['psnr']:.2f}, SSIM: {train_metrics['ssim']:.4f}")
            print("\nValidation Metrics:")
            print(f"G_loss: {val_metrics['g_loss']:.4f}, D_loss: {val_metrics['d_loss']:.4f}")
            print(f"PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours")
        self.writer.close()

def main():
    """Main function to set up and run training."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Dataset paths
    data_dir = 'D:\Wajahat Ali Khan\KHU Gangdong Hospital Data\PET1_FDG\ADNI_Copy'  # Replace with your actual path
    
    # Create datasets
    train_dataset = BrainModalityGANDataset(data_dir, is_train=True)
    val_dataset = BrainModalityGANDataset(data_dir, is_train=False)
    
    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=2,  # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    generator = UNet3D(in_channels=1, out_channels=1, init_features=32)
    discriminator = Discriminator3D()
    
    # Create trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=50,
        save_dir='3d_pix2pix_results_after_long_time_50Epochs'
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred during training process: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        raise