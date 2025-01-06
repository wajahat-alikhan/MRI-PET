import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import nibabel as nib
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DDPMBrainDataset(Dataset):
    def __init__(self, data_dir: str, window_size: int = 5):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
            
        logger.info(f"Scanning directory: {self.data_dir}")
        self.pairs = self.find_pairs()
        
        if len(self.pairs) == 0:
            raise ValueError("No valid MRI-PET pairs found in the directory")
            
        logger.info(f"Successfully initialized dataset with {len(self.pairs)} pairs")

    def find_pairs(self):
        pairs = []
        nii_files = list(self.data_dir.glob("*.nii"))
        logger.info(f"Found {len(nii_files)} total .nii files")
        
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
                logger.debug(f"Found MRI for {key}")
            elif 'PET_FDG_Coreg_Ave' in file_path.name:
                if key not in pairs_dict:
                    pairs_dict[key] = {'mri': None, 'pet': None}
                pairs_dict[key]['pet'] = file_path
                logger.debug(f"Found PET for {key}")
        
        for key, pair_dict in pairs_dict.items():
            if pair_dict['mri'] and pair_dict['pet']:
                pairs.append((pair_dict['mri'], pair_dict['pet']))
                logger.debug(f"Created pair for {key}")
        
        return pairs

    def load_and_preprocess(self, file_path: Path) -> torch.Tensor:
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            
            if np.isnan(data).any():
                median = np.nanmedian(data)
                data = np.nan_to_num(data, nan=median)
            
            # Get center and window indices
            z_dim = data.shape[2]
            center_idx = z_dim // 2
            half_window = self.window_size // 2
            
            # Create three channels with different window averages
            channels = []
            for offset in [-1, 0, 1]:
                center = center_idx + offset
                start = max(0, center - half_window)
                end = min(z_dim, center + half_window + 1)
                
                # Extract and weight slices
                window = data[:, :, start:end]
                weights = np.exp(-0.5 * np.square(np.linspace(-half_window, half_window, end - start)))
                weights = weights / np.sum(weights)
                weighted = np.sum(window * weights.reshape(1, 1, -1), axis=2)
                
                # Convert to tensor and resize
                channel = torch.FloatTensor(weighted)
                channel = F.interpolate(
                    channel.unsqueeze(0).unsqueeze(0),
                    size=(256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                
                # Normalize channel
                channel = (channel - channel.min()) / (channel.max() - channel.min())
                channel = 2 * channel - 1
                channels.append(channel)
            
            # Stack channels into final tensor
            final_tensor = torch.stack(channels)
            return final_tensor
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mri_path, pet_path = self.pairs[idx]
        try:
            mri_tensor = self.load_and_preprocess(mri_path)
            pet_tensor = self.load_and_preprocess(pet_path)
            return mri_tensor, pet_tensor
        except Exception as e:
            logger.error(f"Error loading pair {idx}: {str(e)}")
            raise

class DDPMTrainer:
    def __init__(
        self,
        model,
        scheduler,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=50,
        save_dir='ddpm_checkpoints_cat',
        gradient_clip_val=1.0
    ):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.gradient_clip_val = gradient_clip_val
        
        self.train_losses = []
        self.psnr_scores = []
        self.ssim_scores = []
        self.best_val_loss = float('inf')
        
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=15,
            verbose=True,
            min_lr=1e-6
        )
        
        self.vis_dir = self.save_dir / 'visualizations_cat'
        self.vis_dir.mkdir(exist_ok=True)

    def calculate_metrics(self, pred, target):
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        psnr_vals = []
        ssim_vals = []
        
        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i, 0]
            target_img = target_np[i, 0]
            
            data_range = target_img.max() - target_img.min()
            if data_range == 0:
                continue
                
            psnr_val = psnr(target_img, pred_img, data_range=data_range)
            psnr_vals.append(psnr_val)
            
            ssim_val = ssim(target_img, pred_img, data_range=data_range)
            ssim_vals.append(ssim_val)
        
        if not psnr_vals:
            return 0.0, 0.0
            
        return np.mean(psnr_vals), np.mean(ssim_vals)

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for _, pet in pbar:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pet = pet.to(self.device)
                
                noise = torch.randn_like(pet)
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (pet.shape[0],), device=self.device
                ).long()
                
                noisy_pet = self.scheduler.add_noise(pet, noise, timesteps)
                
                noise_pred = self.model(noisy_pet, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
                
                self.optimizer.step()
                
                with torch.no_grad():
                    denoised = self.scheduler.step(
                        noise_pred, timesteps[0], noisy_pet
                    ).prev_sample
                    psnr_val, ssim_val = self.calculate_metrics(denoised, pet)
                
                epoch_loss += loss.item()
                epoch_psnr += psnr_val
                epoch_ssim += ssim_val
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr_val:.2f}',
                    'ssim': f'{ssim_val:.4f}'
                })
        
        return epoch_loss / num_batches, epoch_psnr / num_batches, epoch_ssim / num_batches

    def train(self):
        logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            try:
                train_loss, train_psnr, train_ssim = self.train_epoch()
                
                self.train_losses.append(train_loss)
                self.psnr_scores.append(train_psnr)
                self.ssim_scores.append(train_ssim)
                
                self.lr_scheduler.step(train_loss)
                
                if train_loss < self.best_val_loss:
                    self.best_val_loss = train_loss
                    self.save_checkpoint(epoch + 1, is_best=True)
                
                if (epoch + 1) % 5 == 0:
                    self.save_prediction_examples(epoch + 1)
                    self.plot_training_curves()
                
                logger.info(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
                logger.info(f"Training Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
                logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch+1}: {str(e)}")
                raise
        
        # Generate final evaluation visualization after training
        logger.info("Creating final evaluation visualization...")
        self.create_final_evaluation_visualization(num_samples=10)
        
        logger.info("Training completed successfully!")

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_losses': self.train_losses,
            'psnr_scores': self.psnr_scores,
            'ssim_scores': self.ssim_scores,
        }
        
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}_cat.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_model_path = self.save_dir / 'best_model_cat.pt'
            torch.save(checkpoint, best_model_path)

    def save_prediction_examples(self, epoch):
        try:
            test_mri, test_pet = next(iter(self.val_loader))
            test_mri = test_mri.to(self.device)
            test_pet = test_pet.to(self.device)
            
            pred_pets = []
            with torch.no_grad():
                for _ in range(5):
                    test_noise = torch.randn_like(test_pet)
                    test_timesteps = torch.zeros(test_pet.shape[0], device=self.device).long()
                    test_noisy = self.scheduler.add_noise(test_pet, test_noise, test_timesteps)
                    test_pred = self.model(test_noisy, test_timesteps).sample
                    pred_pets.append(test_pred)
            
            self.create_prediction_visualization(test_mri, test_pet, pred_pets, epoch)
            
        except Exception as e:
            logger.error(f"Error saving prediction examples: {str(e)}")
            plt.close('all')

    def create_prediction_visualization(self, mri, pet, pred_pets, epoch):
        try:
            mri_img = mri[0, 0].detach().cpu().numpy()
            pet_img = pet[0, 0].detach().cpu().numpy()
            pred_imgs = [pred[0, 0].detach().cpu().numpy() for pred in pred_pets]
            
            vmin = min(pet_img.min(), min(pred.min() for pred in pred_imgs))
            vmax = max(pet_img.max(), max(pred.max() for pred in pred_imgs))
            
            fig, axes = plt.subplots(7, 1, figsize=(15, 25))
            fig.suptitle(f'Predictions at Epoch {epoch} (cat)', fontsize=16, y=0.92)
            
            axes[0].set_title('Input MRI', fontsize=12)
            im0 = axes[0].imshow(mri_img, cmap='gray')
            plt.colorbar(im0, ax=axes[0])
            
            axes[1].set_title('Ground Truth PET', fontsize=12)
            im1 = axes[1].imshow(pet_img, cmap='hot', vmin=vmin, vmax=vmax)
            plt.colorbar(im1, ax=axes[1])
            
            for idx, pred_img in enumerate(pred_imgs):
                axes[idx+2].set_title(f'Generated PET {idx+1}', fontsize=12)
                im = axes[idx+2].imshow(pred_img, cmap='hot', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=axes[idx+2])
            
            for ax in axes:
                ax.axis('off')
            
            plt.tight_layout()
            save_path = self.vis_dir / f'predictions_epoch_{epoch:04d}_cat.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error in prediction visualization: {str(e)}")
            plt.close('all')

    def plot_training_curves(self):
        """Plot and save training metrics over time."""
        try:
            plt.figure(figsize=(15, 10))
            epochs = range(1, len(self.train_losses) + 1)
            
            plt.subplot(2, 1, 1)
            plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time (cat)')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(epochs, self.psnr_scores, 'g-', label='PSNR (dB)')
            plt.plot(epochs, [s * 50 for s in self.ssim_scores], 'r-', 
                    label='SSIM (scaled x50)')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.title('Image Quality Metrics Over Time (cat)')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            save_path = self.save_dir / 'training_metrics_cat.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error in plotting training curves: {str(e)}")
            plt.close('all')

    def create_final_evaluation_visualization(self, num_samples=10):
        """
        Creates a comprehensive visualization comparing model performance across multiple samples.
        This should be called after training is complete.
        """
        save_path = self.save_dir / 'final_evaluation_results_cat.png'
        self.model.eval()
        
        # Create figure with 3 columns (MRI, Real PET, Predicted PET) and num_samples rows
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        fig.suptitle('Model Evaluation: MRI to PET Translation (cat)', fontsize=16, y=0.92)
        
        # Set column titles
        axes[0, 0].set_title('Input MRI', fontsize=14)
        axes[0, 1].set_title('Ground Truth PET', fontsize=14)
        axes[0, 2].set_title('Predicted PET', fontsize=14)
        
        # Get samples from the validation loader
        samples_seen = 0
        val_iter = iter(self.val_loader)
        
        with torch.no_grad():
            while samples_seen < num_samples:
                try:
                    mri, pet = next(val_iter)
                    batch_size = mri.shape[0]
                    
                    for i in range(batch_size):
                        if samples_seen >= num_samples:
                            break
                            
                        # Get single sample
                        single_mri = mri[i:i+1].to(self.device)
                        single_pet = pet[i:i+1].to(self.device)
                        
                        # Generate prediction
                        noise = torch.randn_like(single_pet)
                        timesteps = torch.zeros(1, device=self.device).long()
                        noisy_pet = self.scheduler.add_noise(single_pet, noise, timesteps)
                        pred_pet = self.model(noisy_pet, timesteps).sample
                        
                        # Convert tensors to numpy for visualization
                        mri_np = single_mri[0, 0].cpu().numpy()
                        pet_np = single_pet[0, 0].cpu().numpy()
                        pred_np = pred_pet[0, 0].cpu().numpy()
                        
                        # Determine value range for consistent coloring
                        vmin_pet = min(pet_np.min(), pred_np.min())
                        vmax_pet = max(pet_np.max(), pred_np.max())
                        
                        # Plot MRI
                        im0 = axes[samples_seen, 0].imshow(mri_np, cmap='gray')
                        plt.colorbar(im0, ax=axes[samples_seen, 0])
                        
                        # Plot ground truth PET
                        im1 = axes[samples_seen, 1].imshow(
                            pet_np, 
                            cmap='hot',
                            vmin=vmin_pet,
                            vmax=vmax_pet
                        )
                        plt.colorbar(im1, ax=axes[samples_seen, 1])
                        
                        # Plot predicted PET
                        im2 = axes[samples_seen, 2].imshow(
                            pred_np,
                            cmap='hot',
                            vmin=vmin_pet,
                            vmax=vmax_pet
                        )
                        plt.colorbar(im2, ax=axes[samples_seen, 2])
                        
                        # Calculate and display metrics
                        psnr_val = psnr(pet_np, pred_np, data_range=vmax_pet-vmin_pet)
                        ssim_val = ssim(pet_np, pred_np, data_range=vmax_pet-vmin_pet)
                        
                        # Add metrics as text to the predicted PET plot
                        axes[samples_seen, 2].text(
                            0.02, 0.98,
                            f'PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.4f}',
                            transform=axes[samples_seen, 2].transAxes,
                            verticalalignment='top',
                            color='white',
                            bbox=dict(facecolor='black', alpha=0.7)
                        )
                        
                        # Turn off axes for cleaner look
                        for ax in axes[samples_seen]:
                            ax.axis('off')
                        
                        samples_seen += 1
                
                except StopIteration:
                    break
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Final evaluation visualization saved to {save_path}")

def setup_training(
    data_dir: str,
    batch_size: int = 2,
    num_epochs: int = 50,
    lr: float = 1e-5,
    gradient_clip_val: float = 1.0,
    window_size: int = 5
) -> DDPMTrainer:
    """
    Set up the training process with enhanced error handling and configuration.
    
    Args:
        data_dir: Directory containing the MRI-PET image pairs
        batch_size: Number of samples per batch
        num_epochs: Total number of training epochs
        lr: Initial learning rate
        gradient_clip_val: Maximum gradient norm for clipping
        window_size: Number of slices to consider in the sliding window
    """
    try:
        # Create dataset with sliding window processing
        dataset = DDPMBrainDataset(data_dir, window_size=window_size)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        
        # Ensure reproducibility
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split:], indices[:split]
        
        # Create data samplers and loaders
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=0  # Adjust based on your system
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=0
        )
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model and scheduler
        from diffusers import DDPMPipeline, DDPMScheduler
        
        try:
            pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")
            pipe.to(device)
            scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}")
            raise
        
        # Setup optimizer with improved parameters
        optimizer = torch.optim.AdamW(
            pipe.unet.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Create trainer
        trainer = DDPMTrainer(
            model=pipe.unet,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            gradient_clip_val=gradient_clip_val
        )
        
        return trainer
        
    except Exception as e:
        logger.error(f"Error in training setup: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging to file
    file_handler = logging.FileHandler('training_cat.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    try:
        # Set your data directory
        data_dir = "D:/Wajahat Ali Khan/KHU Gangdong Hospital Data/PET1_FDG/ADNI3_new"
        
        # Create and start trainer
        trainer = setup_training(
            data_dir=data_dir,
            batch_size=2,
            num_epochs=50,
            lr=1e-5,
            gradient_clip_val=1.0,
            window_size=5
        )
        
        # Start training
        logger.info("Initializing training process...")
        trainer.train()
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise