#Testing our finetuned model that we finetuned on ADNI dataset on CelebAHQ dataset, that dataset that it was pretrained on.

import torch
from diffusers import DDPMPipeline
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CelebAHQDataset(Dataset):
    """Custom dataset for CelebAHQ images"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_files = list(self.data_dir.glob('*.jpg'))  # Add more extensions if needed
        
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {data_dir}")
        
        logger.info(f"Found {len(self.image_files)} images in {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

class CelebAHQTester:
    def __init__(self, checkpoint_path, celebahq_data_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        self.celebahq_data_path = Path(celebahq_data_path)
        
        # Initialize model without pretrained weights
        self.pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256", 
                                                safety_checker=None,
                                                requires_safety_checking=False)
        self.load_checkpoint()
        self.pipe.to(self.device)
        
        # Setup data transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def load_checkpoint(self):
        """Load your finetuned checkpoint"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.pipe.unet.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Successfully loaded checkpoint")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def setup_data(self, batch_size=4):
        """Setup CelebAHQ dataset"""
        try:
            dataset = CelebAHQDataset(self.celebahq_data_path, transform=self.transform)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        except Exception as e:
            logger.error(f"Error setting up dataset: {e}")
            raise
    
    def generate_samples(self, num_samples=10):
        """Generate and visualize samples"""
        dataloader = self.setup_data()
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        fig.suptitle('Model Testing on CelebAHQ', fontsize=16)
        
        axes[0, 0].set_title('Original CelebAHQ')
        axes[0, 1].set_title('Noised Image')
        axes[0, 2].set_title('Denoised (Your Model)')
        
        with torch.no_grad():
            for i, images in enumerate(dataloader):
                if i >= num_samples:
                    break
                
                images = images.to(self.device)
                
                # Add noise
                noise = torch.randn_like(images)
                timesteps = torch.zeros(images.shape[0], device=self.device).long()
                noised = self.pipe.scheduler.add_noise(images, noise, timesteps)
                
                # Generate prediction
                prediction = self.pipe.unet(noised, timesteps).sample
                
                # Plot results
                for j in range(images.shape[0]):
                    if i * images.shape[0] + j >= num_samples:
                        break
                    
                    idx = i * images.shape[0] + j
                    
                    # Convert tensors to numpy and denormalize
                    def denorm(x):
                        return (x.cpu().permute(1, 2, 0) * 0.5 + 0.5).numpy()
                    
                    # Plot original
                    axes[idx, 0].imshow(denorm(images[j]))
                    axes[idx, 0].axis('off')
                    
                    # Plot noised
                    axes[idx, 1].imshow(denorm(noised[j]))
                    axes[idx, 1].axis('off')
                    
                    # Plot prediction
                    axes[idx, 2].imshow(denorm(prediction[j]))
                    axes[idx, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('celebahq_test_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Generated test results saved as 'celebahq_test_results.png'")

def main():
    # Update these paths to match your setup
    checkpoint_path = "D:/Wajahat Ali Khan/MRI-PET/ddpm_checkpoints_celebahq/best_model.pt"
    celebahq_data_path = "D:/Wajahat Ali Khan/MRI-PET/ddpm_checkpoints_celebahq/celebahq/celeba_hq_256"
    
    tester = CelebAHQTester(checkpoint_path, celebahq_data_path)
    tester.generate_samples(num_samples=10)

if __name__ == "__main__":
    main()