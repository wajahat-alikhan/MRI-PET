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
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BrainModalityGANDataset(data.Dataset):
    """Dataset class for 3D MRI to PET translation using GAN."""
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
        """Find matching MRI-PET pairs using ADNI filename pattern."""
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
        """Load and preprocess NIfTI files with proper normalization."""
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=np.nanmedian(data))
            
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

class DownSample3D(nn.Module):
    """3D downsampling block."""
    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super(DownSample3D, self).__init__()
        layers = []
        layers.append(
            nn.Conv3d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
        )
        
        if apply_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class UpSample3D(nn.Module):
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super(UpSample3D, self).__init__()
        layers = []
        
        # Transposed convolution
        layers.append(
            nn.ConvTranspose3d(
                in_channels,       # Input channels before concatenation
                out_channels,      # Output channels
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        
        # Batch normalization
        layers.append(nn.BatchNorm3d(out_channels))
        
        # Optional dropout
        if apply_dropout:
            layers.append(nn.Dropout3d(0.5))
        
        # ReLU activation
        layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class Generator3D(nn.Module):
    """
    3D Generator architecture adapted from Pix2Pix for MRI-to-PET synthesis.
    Uses an encoder-decoder structure with skip connections and proper size adjustment
    for 3D medical volumes.
    """
    def __init__(self, input_channels=1, output_channels=1, ngf=64):
        """
        Initialize the Generator with specified channel configurations.
        Args:
            input_channels: Number of input channels (1 for MRI)
            output_channels: Number of output channels (1 for PET)
            ngf: Number of generator filters in first conv layer
        """
        super(Generator3D, self).__init__()

        # Encoder path - each step reduces spatial dimensions by half and increases channels
        self.e1 = nn.Conv3d(input_channels, ngf, 4, stride=2, padding=1)  # First layer without batch norm
        self.e2 = self._make_encoder_block(ngf, ngf * 2)      # 64 -> 128 channels
        self.e3 = self._make_encoder_block(ngf * 2, ngf * 4)  # 128 -> 256 channels
        self.e4 = self._make_encoder_block(ngf * 4, ngf * 8)  # 256 -> 512 channels
        self.e5 = self._make_encoder_block(ngf * 8, ngf * 8)  # 512 -> 512 channels (bottleneck)

        # Decoder path - each step doubles spatial dimensions
        self.d4 = self._make_decoder_block(ngf * 8, ngf * 8)         # 512 -> 512 channels
        self.d3 = self._make_decoder_block(ngf * 8 * 2, ngf * 4)     # 1024 -> 256 channels
        self.d2 = self._make_decoder_block(ngf * 4 * 2, ngf * 2)     # 512 -> 128 channels
        self.d1 = self._make_decoder_block(ngf * 2 * 2, ngf)         # 256 -> 64 channels
        
        # Final layer to generate output
        self.final = nn.Sequential(
            nn.ConvTranspose3d(ngf * 2, output_channels, 4, 2, 1),   # 128 -> 1 channel
            nn.Tanh()
        )

    def _make_encoder_block(self, in_channels, out_channels):
        """Create an encoder block with conv, batch norm, and LeakyReLU."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _make_decoder_block(self, in_channels, out_channels, dropout=False):
        """Create a decoder block with transposed conv, batch norm, and ReLU."""
        layers = [
            nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.insert(2, nn.Dropout3d(0.5))
        return nn.Sequential(*layers)

    def _adjust_size(self, x, target_tensor=None, target_size=None):
        """
        Adjust tensor size to match target through cropping or padding.
        Adapted from 3D U-Net implementation for proper size handling.
        """
        if target_tensor is not None:
            target_size = target_tensor.size()
        
        if x.size() == target_size:
            return x
            
        pad_dims = []
        for i, (src, dst) in enumerate(zip(x.shape[2:], target_size[2:])):
            pad_amt = dst - src
            if pad_amt > 0:
                # Need to pad
                pad_dims.extend([pad_amt//2, pad_amt - pad_amt//2])
            elif pad_amt < 0:
                # Need to crop
                crop = -pad_amt
                x = x.narrow(i+2, crop//2, src - crop)
                pad_dims.extend([0, 0])
            else:
                pad_dims.extend([0, 0])
                
        if any(p != 0 for p in pad_dims):
            x = F.pad(x, pad_dims)
            
        return x

    def forward(self, x):
        """
        Forward pass with size-adjusted skip connections.
        Includes size tracking prints for debugging.
        """
        # Store input tensor for final size adjustment
        input_tensor = x
        
        # Encoder path
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        bottleneck = self.e5(e4)

        # Decoder path with skip connections
        d4 = self.d4(bottleneck)
        adjusted_e4 = self._adjust_size(e4, target_tensor=d4)
        d4 = torch.cat([d4, adjusted_e4], 1)
        d3 = self.d3(d4)
        adjusted_e3 = self._adjust_size(e3, target_tensor=d3)
        d3 = torch.cat([d3, adjusted_e3], 1)
        d2 = self.d2(d3)
        adjusted_e2 = self._adjust_size(e2, target_tensor=d2)
        d2 = torch.cat([d2, adjusted_e2], 1)
        d1 = self.d1(d2)
        adjusted_e1 = self._adjust_size(e1, target_tensor=d1)
        d1 = torch.cat([d1, adjusted_e1], 1)
        
        # Final layer and size adjustment
        output = self.final(d1)
        output = self._adjust_size(output, target_tensor=input_tensor)
        
        return output

class Discriminator3D(nn.Module):
    """3D PatchGAN discriminator adapted for medical volume size.
    The architecture follows the same progression as the generator but in reverse,
    creating a receptive field appropriate for 3D patch discrimination."""
    
    def __init__(self, input_channels=2, ndf=64):
        """
        Args:
            input_channels: Combined input channels (1 for MRI + 1 for PET = 2)
            ndf: Base number of discriminator filters, analogous to ngf in generator
        """
        super(Discriminator3D, self).__init__()

        # Initial layer without normalization, following original PatchGAN
        # Input: 2 x 79 x 95 x 79 (concatenated MRI and PET)
        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )  # Output: ndf x 39 x 47 x 39

        # Main discriminator layers with increasing channel depth
        # We use 4 layers to match our generator's depth
        self.layer1 = self._make_layer(ndf, ndf * 2)      # ndf*2 x 19 x 23 x 19
        self.layer2 = self._make_layer(ndf * 2, ndf * 4)  # ndf*4 x 9 x 11 x 9
        self.layer3 = self._make_layer(ndf * 4, ndf * 8)  # ndf*8 x 4 x 5 x 4

        # Final classification layer
        # Instead of reducing to 1x1x1, we maintain a small patch output
        self.final = nn.Conv3d(ndf * 8, 1, 4, stride=1, padding=1)

    def _make_layer(self, in_channels, out_channels):
        """Creates a discriminator layer following PatchGAN pattern with 3D operations."""
        return nn.Sequential(
            # Strided convolution for downsampling
            nn.Conv3d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            # Batch normalization helps training stability
            nn.BatchNorm3d(out_channels),
            # LeakyReLU prevents dead gradients
            nn.LeakyReLU(0.2)
        )

    def forward(self, mri, pet):
        """
        Forward pass concatenating MRI and PET volumes.
        Returns spatial patches indicating real/fake predictions for different regions.
        """
        # Concatenate MRI and PET along channel dimension
        x = torch.cat([mri, pet], dim=1)
        
        # Sequential feature extraction and downsampling
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Final classification preserving spatial patches
        return self.final(x)

def calculate_metrics(pred, target):
    """Calculate PSNR and SSIM metrics."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    psnr_vals = []
    ssim_vals = []
    
    for i in range(pred_np.shape[0]):
        # Calculate metrics for middle slices in each dimension
        z_slice = pred_np.shape[2] // 2
        
        pred_slice = pred_np[i, 0, :, :, z_slice]
        target_slice = target_np[i, 0, :, :, z_slice]
        
        # Calculate PSNR
        psnr_val = psnr(target_slice, pred_slice, 
                       data_range=target_slice.max() - target_slice.min())
        psnr_vals.append(psnr_val)
        
        # Calculate SSIM
        ssim_val = ssim(target_slice, pred_slice,
                       data_range=target_slice.max() - target_slice.min())
        ssim_vals.append(ssim_val)
    
    return np.mean(psnr_vals), np.mean(ssim_vals)

def generator_loss(disc_generated_output, gen_output, target, lambda_l1=50):
    """Calculate generator loss including adversarial and L1 components."""
    loss_object = nn.BCEWithLogitsLoss()
    gan_loss = loss_object(
        disc_generated_output,
        torch.ones_like(disc_generated_output)
    )
    
    l1_loss = torch.mean(torch.abs(target - gen_output))
    total_gen_loss = gan_loss + (lambda_l1 * l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """Calculate discriminator loss."""
    loss_object = nn.BCEWithLogitsLoss()
    
    real_loss = loss_object(
        disc_real_output,
        torch.ones_like(disc_real_output)
    )
    generated_loss = loss_object(
        disc_generated_output,
        torch.zeros_like(disc_generated_output)
    )
    
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

class GANTrainer:
    """Trainer class for 3D Pix2Pix GAN."""
    def __init__(self, generator, discriminator, train_loader, val_loader,
                 device, num_epochs=200, save_dir='3d_pix2pix_checkpoints'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        
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
        
        # Get middle slices
        z_slice = mri.shape[2] // 2
        y_slice = mri.shape[3] // 2
        x_slice = mri.shape[4] // 2
        
        for batch_idx in range(min(mri.shape[0], 4)):  # Save up to 4 examples
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            
            # Axial view
            axes[0, 0].imshow(mri[batch_idx, 0, :, :, z_slice].cpu(), cmap='gray')
            axes[0, 0].set_title('MRI')
            axes[0, 1].imshow(pred_pet[batch_idx, 0, :, :, z_slice].cpu(), cmap='hot')
            axes[0, 1].set_title('Generated PET')
            axes[0, 2].imshow(true_pet[batch_idx, 0, :, :, z_slice].cpu(), cmap='hot')
            axes[0, 2].set_title('True PET')
            
            # Coronal view
            axes[1, 0].imshow(mri[batch_idx, 0, :, y_slice, :].cpu(), cmap='gray')
            axes[1, 1].imshow(pred_pet[batch_idx, 0, :, y_slice, :].cpu(), cmap='hot')
            axes[1, 2].imshow(true_pet[batch_idx, 0, :, y_slice, :].cpu(), cmap='hot')
            
            # Sagittal view
            axes[2, 0].imshow(mri[batch_idx, 0, x_slice, :, :].cpu(), cmap='gray')
            axes[2, 1].imshow(pred_pet[batch_idx, 0, x_slice, :, :].cpu(), cmap='hot')
            axes[2, 2].imshow(true_pet[batch_idx, 0, x_slice, :, :].cpu(), cmap='hot')
            
            plt.tight_layout()
            plt.savefig(save_dir / f'epoch_{epoch}_sample_{batch_idx}.png')
            plt.close()

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

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        num_batches = len(self.train_loader)
        
        for i, (mri, pet) in enumerate(self.train_loader):
            mri, pet = mri.to(self.device), pet.to(self.device)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            generated_pet = self.generator(mri)
            
            real_output = self.discriminator(mri, pet)
            fake_output = self.discriminator(mri, generated_pet.detach())
            
            d_loss = discriminator_loss(real_output, fake_output)
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            fake_output = self.discriminator(mri, generated_pet)
            g_total_loss, g_gan_loss, g_l1_loss = generator_loss(fake_output, generated_pet, pet)
            
            g_total_loss.backward()
            self.g_optimizer.step()
            
            # Calculate metrics
            psnr_val, ssim_val = calculate_metrics(generated_pet, pet)
            
            # Update running averages
            epoch_g_loss += g_total_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_psnr += psnr_val
            epoch_ssim += ssim_val
            
            if i % 10 == 0:
                print(f'Epoch [{epoch}/{self.num_epochs}] Batch [{i}/{num_batches}] '
                      f'G_loss: {g_total_loss.item():.4f} D_loss: {d_loss.item():.4f} '
                      f'PSNR: {psnr_val:.2f} SSIM: {ssim_val:.4f}')
        
        # Calculate epoch averages
        metrics = {
            'g_loss': epoch_g_loss / num_batches,
            'd_loss': epoch_d_loss / num_batches,
            'psnr': epoch_psnr / num_batches,
            'ssim': epoch_ssim / num_batches
        }
        
        return metrics

    def validate(self, epoch):
        """Run validation."""
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
                
                # Generate fake PET
                generated_pet = self.generator(mri)
                
                # Calculate discriminator outputs
                real_output = self.discriminator(mri, pet)
                fake_output = self.discriminator(mri, generated_pet)
                
                # Calculate losses
                d_loss = discriminator_loss(real_output, fake_output)
                g_total_loss, _, _ = generator_loss(fake_output, generated_pet, pet)
                
                # Calculate metrics
                psnr_val, ssim_val = calculate_metrics(generated_pet, pet)
                
                val_g_loss += g_total_loss.item()
                val_d_loss += d_loss.item()
                val_psnr += psnr_val
                val_ssim += ssim_val
                
                # Save examples from first batch
                if i == 0:
                    self.save_prediction_examples(mri, generated_pet, pet, epoch)
        
        # Calculate validation averages
        metrics = {
            'g_loss': val_g_loss / num_batches,
            'd_loss': val_d_loss / num_batches,
            'psnr': val_psnr / num_batches,
            'ssim': val_ssim / num_batches
        }
        
        return metrics

    def train(self):
        """Main training loop."""
        print("Starting training...")
        start_time = time.time()
        best_val_psnr = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_metrics = self.train_epoch(epoch)
            for key, value in train_metrics.items():
                self.train_metrics[key].append(value)
            
            # Validation phase
            val_metrics = self.validate(epoch)
            for key, value in val_metrics.items():
                self.val_metrics[key].append(value)
            
            # Log metrics to TensorBoard
            for key in train_metrics:
                self.writer.add_scalar(f'train/{key}', train_metrics[key], epoch)
                self.writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
            
            # Save best model
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
            print(f"\nEpoch {epoch} Summary:")
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
        batch_size=10,  # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    generator = Generator3D()
    discriminator = Discriminator3D()
    
    # Create trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=151,
        save_dir='3d_pix2pix_results'
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise