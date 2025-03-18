#changing our unet model to the generator of pix2pix.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, SubsetRandomSampler
import nibabel as nib
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
from torch.utils.tensorboard import SummaryWriter

# Constants
LAMBDA = 100  # L1 loss weight in generator loss
BATCH_SIZE = 4  # Reduced batch size due to 3D volumes
IMG_DEPTH = 128  # Adjust based on your typical volume size
IMG_HEIGHT = 128
IMG_WIDTH = 128

class DoubleConv3D(nn.Module):
    """Double 3D convolution block"""
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

class Generator3D(nn.Module):
    """3D Generator with U-Net architecture"""
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(Generator3D, self).__init__()
        
        features = init_features
        
        # Encoder path
        self.encoder1 = DoubleConv3D(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = DoubleConv3D(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = DoubleConv3D(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bridge
        self.bridge = DoubleConv3D(features * 4, features * 8)
        
        # Decoder path
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, 
                                         kernel_size=2, stride=2)
        self.decoder3 = DoubleConv3D(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2,
                                         kernel_size=2, stride=2)
        self.decoder2 = DoubleConv3D(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features,
                                         kernel_size=2, stride=2)
        self.decoder1 = DoubleConv3D(features * 2, features)
        
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        self.final_act = nn.Tanh()
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool3(enc3)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder with skip connections
        x = self.upconv3(x)
        x = self._adjust_size(x, enc3)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = self._adjust_size(x, enc2)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = self._adjust_size(x, enc1)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        
        x = self.final_conv(x)
        x = self.final_act(x)
        
        return x
    
    def _adjust_size(self, x, target_tensor):
        """Adjust tensor sizes for skip connections"""
        if x.size()[2:] != target_tensor.size()[2:]:
            x = F.interpolate(x, size=target_tensor.size()[2:], 
                            mode='trilinear', align_corners=False)
        return x

class Discriminator3D(nn.Module):
    """3D PatchGAN discriminator"""
    def __init__(self):
        super(Discriminator3D, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1: input is (nc) x 128 x 128 x 128
            nn.Conv3d(2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: (64) x 64 x 64 x 64
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: (128) x 32 x 32 x 32
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: (256) x 16 x 16 x 16
            nn.Conv3d(256, 512, 4, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: (512) x 15 x 15 x 15
            nn.Conv3d(512, 1, 4, stride=1, padding=1)
        )
        
    def forward(self, mri, pet):
        # Concatenate input and output channels
        x = torch.cat([mri, pet], dim=1)
        return self.model(x)

class BrainModalityDataset(data.Dataset):
    """Dataset class for MRI-PET pairs"""
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
        
        print(f"Scanning directory: {self.data_dir}")
        self.pairs = self._find_pairs()
        
        if len(self.pairs) == 0:
            raise ValueError("No valid MRI-PET pairs found in the directory")
        
        print(f"Successfully initialized dataset with {len(self.pairs)} pairs")
    
    def _find_pairs(self) -> list:
        pairs = []
        nii_files = list(self.data_dir.glob("*.nii"))
        print(f"Found {len(nii_files)} total .nii files")
        
        pairs_dict = {}
        for file_path in nii_files:
            pattern = r'(\d+_S_\d+)_(\d{4}[-_]\d{2})'
            match = re.search(pattern, file_path.name)
            if not match:
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
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # Handle NaN values
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=np.nanmedian(data))
            
            # Normalize to [-1, 1] range for GAN training
            data_min, data_max = np.min(data), np.max(data)
            data = (data - data_min) / (data_max - data_min)
            data = data * 2 - 1
            
            # Add channel dimension
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
        
        if self.transform:
            mri_tensor = self.transform(mri_tensor)
            pet_tensor = self.transform(pet_tensor)
        
        return mri_tensor, pet_tensor

class Pix2PixTrainer:
    """Trainer class for 3D Pix2Pix"""
    def __init__(self, generator, discriminator, train_loader, val_loader, 
                 device, num_epochs=150, save_dir='3d_Unet_pix2pix_checkpoints'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create directories for visualizations
        self.vis_dir = self.save_dir / 'visualizations'
        self.vis_dir.mkdir(exist_ok=True)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(f'{save_dir}/logs')
        
        # Initialize metric tracking
        self.train_g_losses = []
        self.train_d_losses = []
        self.val_g_losses = []
        self.val_d_losses = []
        self.psnr_scores = []
        self.ssim_scores = []
    
    def save_volume_samples(self, mri, pred_pet, true_pet, epoch):
        """Save central slices from different views of the volume"""
        # Take first sample from batch
        mri = mri[0].cpu().numpy()
        pred_pet = pred_pet[0].cpu().numpy()
        true_pet = true_pet[0].cpu().numpy()
        
        # Get central slices
        d, h, w = mri.shape[1:]
        
        # Create figure with three rows (one for each modality) and three columns (one for each view)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Plot each modality and view
        for row, (data, title) in enumerate(zip(
            [mri[0], pred_pet[0], true_pet[0]], 
            ['MRI', 'Predicted PET', 'True PET'])):
            
            # Sagittal view (middle slice in x-axis)
            axes[row, 0].imshow(data[w//2, :, :], cmap='gray')
            axes[row, 0].set_title(f'{title} - Sagittal')
            axes[row, 0].axis('off')
            
            # Coronal view (middle slice in y-axis)
            axes[row, 1].imshow(data[:, h//2, :], cmap='gray')
            axes[row, 1].set_title(f'{title} - Coronal')
            axes[row, 1].axis('off')
            
            # Axial view (middle slice in z-axis)
            axes[row, 2].imshow(data[:, :, d//2], cmap='gray')
            axes[row, 2].set_title(f'{title} - Axial')
            axes[row, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / f'epoch_{epoch}_samples.png')
        plt.close()
    
    def calculate_metrics(self, pred, target):
        """Calculate PSNR and SSIM for the central slices"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Calculate metrics for central slices
        d = pred_np.shape[2] // 2
        
        psnr_val = psnr(target_np[0, 0, :, :, d], pred_np[0, 0, :, :, d], 
                        data_range=2.0)  # range is 2.0 because of [-1, 1] normalization
        ssim_val = ssim(target_np[0, 0, :, :, d], pred_np[0, 0, :, :, d],
                        data_range=2.0)
        
        return psnr_val, ssim_val
    
    def generator_loss(self, disc_generated_output, gen_output, target):
        """Calculate generator loss combining adversarial and L1 loss"""
        loss_object = nn.BCEWithLogitsLoss()
        gan_loss = loss_object(
            disc_generated_output,
            torch.ones_like(disc_generated_output)
        )
        
        # L1 loss between generated and target PET
        l1_loss = torch.mean(torch.abs(target - gen_output))
        
        # Total generator loss
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        
        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """Calculate discriminator loss"""
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

    def train_step(self, mri, real_pet):
        """Perform one training step"""
        # Generate fake PET
        fake_pet = self.generator(mri)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real discriminator output
        disc_real_output = self.discriminator(mri, real_pet)
        # Fake discriminator output
        disc_generated_output = self.discriminator(mri, fake_pet.detach())
        
        d_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Get new discriminator outputs for generator training
        disc_generated_output = self.discriminator(mri, fake_pet)
        
        g_total_loss, g_gan_loss, g_l1_loss = self.generator_loss(
            disc_generated_output, fake_pet, real_pet)
        
        g_total_loss.backward()
        self.g_optimizer.step()
        
        return {
            'g_total_loss': g_total_loss.item(),
            'g_gan_loss': g_gan_loss.item(),
            'g_l1_loss': g_l1_loss.item(),
            'd_loss': d_loss.item(),
            'fake_pet': fake_pet
        }

    def train(self):
        """Main training loop"""
        print("Starting training...")
        start_time = time.time()
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.generator.train()
            self.discriminator.train()
            
            train_g_loss = 0
            train_d_loss = 0
            
            with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}') as pbar:
                for batch_idx, (mri, real_pet) in enumerate(pbar):
                    mri = mri.to(self.device)
                    real_pet = real_pet.to(self.device)
                    
                    # Training step
                    results = self.train_step(mri, real_pet)
                    
                    train_g_loss += results['g_total_loss']
                    train_d_loss += results['d_loss']
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'G_loss': f"{results['g_total_loss']:.4f}",
                        'D_loss': f"{results['d_loss']:.4f}"
                    })
                    
                    # Log to tensorboard
                    step = epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('Train/G_loss', results['g_total_loss'], step)
                    self.writer.add_scalar('Train/D_loss', results['d_loss'], step)
            
            # Calculate average epoch losses
            train_g_loss /= len(self.train_loader)
            train_d_loss /= len(self.train_loader)
            
            # Validation phase
            self.generator.eval()
            self.discriminator.eval()
            
            val_g_loss = 0
            val_d_loss = 0
            val_psnr = 0
            val_ssim = 0
            
            with torch.no_grad():
                for batch_idx, (mri, real_pet) in enumerate(self.val_loader):
                    mri = mri.to(self.device)
                    real_pet = real_pet.to(self.device)
                    
                    # Generate fake PET
                    fake_pet = self.generator(mri)
                    
                    # Calculate discriminator outputs
                    disc_real_output = self.discriminator(mri, real_pet)
                    disc_generated_output = self.discriminator(mri, fake_pet)
                    
                    # Calculate losses
                    g_loss, _, _ = self.generator_loss(
                        disc_generated_output, fake_pet, real_pet)
                    d_loss = self.discriminator_loss(
                        disc_real_output, disc_generated_output)
                    
                    val_g_loss += g_loss.item()
                    val_d_loss += d_loss.item()
                    
                    # Calculate metrics
                    psnr_val, ssim_val = self.calculate_metrics(fake_pet, real_pet)
                    val_psnr += psnr_val
                    val_ssim += ssim_val
                    
                    # Save example images
                    if batch_idx == 0:
                        self.save_volume_samples(mri, fake_pet, real_pet, epoch)
            
            # Calculate average validation metrics
            val_g_loss /= len(self.val_loader)
            val_d_loss /= len(self.val_loader)
            val_psnr /= len(self.val_loader)
            val_ssim /= len(self.val_loader)
            
            # Log validation metrics
            self.writer.add_scalar('Val/G_loss', val_g_loss, epoch)
            self.writer.add_scalar('Val/D_loss', val_d_loss, epoch)
            self.writer.add_scalar('Val/PSNR', val_psnr, epoch)
            self.writer.add_scalar('Val/SSIM', val_ssim, epoch)
            
            # Save best model
            if val_g_loss < best_val_loss:
                best_val_loss = val_g_loss
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                    'val_loss': val_g_loss,
                }, self.save_dir / 'best_model.pt')
            
            # Regular checkpoint saving
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                }, self.save_dir / f'checkpoint_epoch_{epoch+1}.pt')
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train - G_loss: {train_g_loss:.4f}, D_loss: {train_d_loss:.4f}")
            print(f"Val - G_loss: {val_g_loss:.4f}, D_loss: {val_d_loss:.4f}")
            print(f"Metrics - PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        self.writer.close()

def train_model(data_dir, batch_size=4, num_epochs=150, learning_rate=2e-4):
    """Main function to setup and start training"""
    
    # Create dataset
    dataset = BrainModalityDataset(data_dir)
    
    # Create train/val split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    generator = Generator3D(in_channels=1, out_channels=1)
    discriminator = Discriminator3D()
    
    # Create trainer
    trainer = Pix2PixTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs
    )
    
    return trainer

if __name__ == "__main__":
    try:
        # Define data directory and parameters
        data_dir = 'D:\Wajahat Ali Khan\KHU Gangdong Hospital Data\PET1_FDG\ADNI_Copy'  # Replace with your data path
        batch_size = 4
        num_epochs = 150
        learning_rate = 2e-4
        
        # Create and start training
        trainer = train_model(
            data_dir=data_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    