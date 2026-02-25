import torch
import math

class AegisEntropyMonitor:
    """
    Aegis Framework PoC: System-Level Alignment Monitor.
    Calculates the Shannon Entropy of latent/visual states to detect 
    Out-of-Distribution (OOD) hallucinations or 'Entropy Collapse' 
    during the diffusion denoising process.
    """
    def __init__(self, threshold_low=3.0, threshold_high=7.5):
        # Entropy bounds. If the image is too blurry/uniform, entropy drops.
        # If it's pure white noise, entropy spikes.
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def check_entropy_collapse(self, frame_tensor: torch.Tensor) -> bool:
        """
        Analyzes a PyTorch tensor frame representing the World Model's output.
        Returns True if a collapse is detected, triggering the sandbox shunt.
        """
        # 1. Convert to grayscale to evaluate structural entropy
        if frame_tensor.dim() == 4:  # Batched (B, C, H, W)
            gray = frame_tensor.mean(dim=1, keepdim=True)
        elif frame_tensor.dim() == 3:  # Single image (C, H, W)
            gray = frame_tensor.mean(dim=0, keepdim=True)
        else:
            return False

        # 2. Normalize and discretize into 256 bins (simulating pixel values)
        # We clamp to ensure values are within [0, 1]
        gray = torch.clamp(gray, 0.0, 1.0)
        bins = torch.histc(gray, bins=256, min=0.0, max=1.0)
        
        # 3. Calculate probability distribution of pixel intensities
        p = bins / bins.sum()
        p = p[p > 0]  # Avoid log(0)
        
        # 4. Shannon Entropy Formula: H = -sum(p * log2(p))
        entropy = -(p * torch.log2(p)).sum().item()
        
        # 5. Monitor and Alert
        # print(f"[Aegis Diagnostics] Current Frame Entropy: {entropy:.4f}")
        
        if entropy < self.threshold_low:
            print(f"\n⚠️ [AEGIS KERNEL ALERT] Entropy Collapse Detected ({entropy:.2f} < {self.threshold_low})")
            print("⚠️ [AEGIS] Halting state-transition. Shunting agent to gVisor ephemeral sandbox...")
            return True
            
        return False
