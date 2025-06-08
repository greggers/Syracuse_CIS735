
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct, idct

def hide_image(cover_img, secret_img, bits_to_use=2):
    # Convert images to numpy arrays
    cover = np.array(cover_img)
    secret = np.array(secret_img)
    
    # Resize secret image to match cover image
    secret = Image.fromarray(secret).resize(cover_img.size)
    secret = np.array(secret)
    
    # Create output array
    stego = cover.copy()
    
    # Clear the lowest bits_to_use bits of the cover image
    stego = stego & (255 << bits_to_use)
    
    # Scale down secret image and shift it to the right position
    secret = secret >> (8 - bits_to_use)
    
    # Combine images
    stego = stego | secret
    
    return stego

def extract_image(stego_img, bits_used=2):
    # Extract the hidden image
    stego = np.array(stego_img)
    
    # Mask and shift to get original values
    extracted = (stego & ((1 << bits_used) - 1)) << (8 - bits_used)
    
    return extracted

def dct2(block):
    """
    2D Discrete Cosine Transform
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """
    2D Inverse Discrete Cosine Transform
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def hide_image_dct(cover_img, secret_img, alpha=0.1, block_size=8):
    """
    Hide secret image in cover image using DCT (Discrete Cosine Transform) steganography
    
    Parameters:
    -----------
    cover_img : PIL Image
        Cover image to hide data in
    secret_img : PIL Image
        Secret image to hide
    alpha : float
        Embedding strength (0.01-0.5, lower = more subtle)
    block_size : int
        Size of DCT blocks (typically 8x8)
    
    Returns:
    --------
    numpy.ndarray : Steganographic image with hidden data
    """
    # Convert images to numpy arrays
    cover = np.array(cover_img, dtype=np.float32)
    secret = np.array(secret_img, dtype=np.float32)
    
    # Resize secret image to match cover image
    secret = np.array(Image.fromarray(secret.astype(np.uint8)).resize(cover_img.size), dtype=np.float32)
    
    # Normalize images to [0, 1]
    cover = cover / 255.0
    secret = secret / 255.0
    
    # Get image dimensions
    height, width = cover.shape
    
    # Pad images to be divisible by block_size
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    
    cover_padded = np.pad(cover, ((0, pad_height), (0, pad_width)), mode='edge')
    secret_padded = np.pad(secret, ((0, pad_height), (0, pad_width)), mode='edge')
    
    # Initialize output image
    stego = cover_padded.copy()
    
    # Process image in blocks
    for i in range(0, cover_padded.shape[0], block_size):
        for j in range(0, cover_padded.shape[1], block_size):
            # Extract blocks
            cover_block = cover_padded[i:i+block_size, j:j+block_size]
            secret_block = secret_padded[i:i+block_size, j:j+block_size]
            
            # Apply DCT to both blocks
            cover_dct = dct2(cover_block)
            secret_dct = dct2(secret_block)
            
            # Embed secret in mid-frequency coefficients (more robust than high-frequency)
            # Avoid DC coefficient (0,0) and high-frequency coefficients
            stego_dct = cover_dct.copy()
            
            # Embed in selected DCT coefficients
            for u in range(1, min(4, block_size)):  # Mid-frequency range
                for v in range(1, min(4, block_size)):
                    if u + v > 1 and u + v < 6:  # Select mid-frequency coefficients
                        # Embed secret DCT coefficient into cover DCT coefficient
                        stego_dct[u, v] = cover_dct[u, v] + alpha * secret_dct[u, v]
            
            # Apply inverse DCT
            stego_block = idct2(stego_dct)
            
            # Ensure values are in valid range
            stego_block = np.clip(stego_block, 0, 1)
            
            # Store the modified block
            stego[i:i+block_size, j:j+block_size] = stego_block
    
    # Remove padding and convert back to uint8
    stego = stego[:height, :width]
    stego = (stego * 255).astype(np.uint8)
    
    return stego

def extract_image_dct(stego_img, cover_img, alpha=0.1, block_size=8):
    """
    Extract secret image from steganographic image using DCT method
    
    Parameters:
    -----------
    stego_img : numpy.ndarray or PIL Image
        Steganographic image containing hidden data
    cover_img : PIL Image
        Original cover image
    alpha : float
        Embedding strength used during hiding
    block_size : int
        Size of DCT blocks used during hiding
    
    Returns:
    --------
    numpy.ndarray : Extracted secret image
    """
    # Convert to numpy arrays
    if isinstance(stego_img, Image.Image):
        stego = np.array(stego_img, dtype=np.float32)
    else:
        stego = stego_img.astype(np.float32)
    
    cover = np.array(cover_img, dtype=np.float32)
    
    # Normalize to [0, 1]
    stego = stego / 255.0
    cover = cover / 255.0
    
    # Get dimensions
    height, width = stego.shape
    
    # Pad images
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    
    stego_padded = np.pad(stego, ((0, pad_height), (0, pad_width)), mode='edge')
    cover_padded = np.pad(cover, ((0, pad_height), (0, pad_width)), mode='edge')
    
    # Initialize extracted image
    extracted = np.zeros_like(stego_padded)
    
    # Process in blocks
    for i in range(0, stego_padded.shape[0], block_size):
        for j in range(0, stego_padded.shape[1], block_size):
            # Extract blocks
            stego_block = stego_padded[i:i+block_size, j:j+block_size]
            cover_block = cover_padded[i:i+block_size, j:j+block_size]
            
            # Apply DCT
            stego_dct = dct2(stego_block)
            cover_dct = dct2(cover_block)
            
            # Extract secret DCT coefficients
            secret_dct = np.zeros_like(stego_dct)
            
            for u in range(1, min(4, block_size)):
                for v in range(1, min(4, block_size)):
                    if u + v > 1 and u + v < 6:
                        # Extract the embedded coefficient
                        secret_dct[u, v] = (stego_dct[u, v] - cover_dct[u, v]) / alpha
            
            # Apply inverse DCT to get secret block
            secret_block = idct2(secret_dct)
            secret_block = np.clip(secret_block, 0, 1)
            
            extracted[i:i+block_size, j:j+block_size] = secret_block
    
    # Remove padding and convert back to uint8
    extracted = extracted[:height, :width]
    extracted = (extracted * 255).astype(np.uint8)
    
    return extracted

def calculate_image_quality_metrics(original, modified):
    """
    Calculate image quality metrics for steganography evaluation
    """
    # Convert to float for calculations
    orig = original.astype(np.float64)
    mod = modified.astype(np.float64)
    
    # Mean Squared Error
    mse = np.mean((orig - mod) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Structural Similarity Index (simplified version)
    mean_orig = np.mean(orig)
    mean_mod = np.mean(mod)
    var_orig = np.var(orig)
    var_mod = np.var(mod)
    cov = np.mean((orig - mean_orig) * (mod - mean_mod))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mean_orig * mean_mod + c1) * (2 * cov + c2)) / \
           ((mean_orig**2 + mean_mod**2 + c1) * (var_orig + var_mod + c2))
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim
    }

def main():
    """
    Main function to demonstrate steganography techniques
    """
    print("="*60)
    print("STEGANOGRAPHY DEMONSTRATION")
    print("Concealment, Camouflage, and Deception (CCD) Methods")
    print("="*60)
    
    print("\nSteganography: The art of hiding information in plain sight")
    print("This demonstration shows two methods of concealing a secret image:")
    print("1. LSB (Least Significant Bit) manipulation")
    print("2. DCT (Discrete Cosine Transform) frequency domain hiding\n")
    
    try:
        # Load images
        print("Loading images...")
        cover_img = Image.open('Lecture_9_CCD_Methods/sample_images/cover.jpg').convert('L')
        secret_img = Image.open('Lecture_9_CCD_Methods/sample_images/secret.jpg').convert('L')
        
        print(f"Cover image size: {cover_img.size}")
        print(f"Secret image size: {secret_img.size}")
        
        # LSB Steganography
        print("\n" + "="*50)
        print("LSB STEGANOGRAPHY")
        print("="*50)
        print("Creating LSB steganographic images...")
        
        stego_2bit = hide_image(cover_img, secret_img, bits_to_use=2)
        stego_4bit = hide_image(cover_img, secret_img, bits_to_use=4)
        
        # Extract hidden images (LSB)
        extracted_2bit = extract_image(stego_2bit, bits_used=2)
        extracted_4bit = extract_image(stego_4bit, bits_used=4)
        
        # DCT Steganography
        print("\n" + "="*50)
        print("DCT STEGANOGRAPHY")
        print("="*50)
        print("Creating DCT steganographic images...")
        
        # Different alpha values for DCT
        stego_dct_low = hide_image_dct(cover_img, secret_img, alpha=0.05)   # Very subtle
        stego_dct_high = hide_image_dct(cover_img, secret_img, alpha=0.15)  # More visible
        
        # Extract hidden images (DCT)
        extracted_dct_low = extract_image_dct(stego_dct_low, cover_img, alpha=0.05)
        extracted_dct_high = extract_image_dct(stego_dct_high, cover_img, alpha=0.15)
        
        # Display results
        print("\nGenerating comprehensive visualization...")
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Row 1: Original images and LSB results
        axes[0, 0].imshow(cover_img, cmap='gray')
        axes[0, 0].set_title('Original Cover Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(secret_img, cmap='gray')
        axes[0, 1].set_title('Secret Image')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(stego_2bit, cmap='gray')
        axes[0, 2].set_title('LSB Stego (2 bits)\nSubtle concealment')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(stego_4bit, cmap='gray')
        axes[0, 3].set_title('LSB Stego (4 bits)\nMore obvious distortion')
        axes[0, 3].axis('off')
        
        # Row 2: DCT results
        axes[1, 0].imshow(stego_dct_low, cmap='gray')
        axes[1, 0].set_title('DCT Stego (α=0.05)\nFrequency domain hiding')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(stego_dct_high, cmap='gray')
        axes[1, 1].set_title('DCT Stego (α=0.15)\nStronger embedding')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(extracted_2bit, cmap='gray')
        axes[1, 2].set_title('LSB Extracted (2 bits)')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(extracted_4bit, cmap='gray')
        axes[1, 3].set_title('LSB Extracted (4 bits)')
        axes[1, 3].axis('off')
        
        # Row 3: DCT extractions and difference images
        axes[2, 0].imshow(extracted_dct_low, cmap='gray')
        axes[2, 0].set_title('DCT Extracted (α=0.05)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(extracted_dct_high, cmap='gray')
        axes[2, 1].set_title('DCT Extracted (α=0.15)')
        axes[2, 1].axis('off')
        
        # Difference images to show distortion
        diff_lsb = np.abs(np.array(cover_img).astype(float) - stego_2bit.astype(float))
        diff_dct = np.abs(np.array(cover_img).astype(float) - stego_dct_low.astype(float))
        
        axes[2, 2].imshow(diff_lsb, cmap='hot')
        axes[2, 2].set_title('LSB Distortion\n(Amplified 10x)')
        axes[2, 2].axis('off')
        
        axes[2, 3].imshow(diff_dct, cmap='hot')
        axes[2, 3].set_title('DCT Distortion\n(Amplified 10x)')
        axes[2, 3].axis('off')
        
        plt.suptitle('Steganography Comparison: LSB vs DCT Methods', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the comprehensive figure
        output_filename = 'Lecture_9_CCD_Methods/steganography_comprehensive_demo.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\nComprehensive visualization saved as: {output_filename}")
        
        plt.show()
        
        # Quality Analysis
        print("\n" + "="*60)
        print("STEGANOGRAPHY QUALITY ANALYSIS")
        print("="*60)
        
        cover_array = np.array(cover_img)
        
        # Calculate quality metrics for all methods
        print("\nLSB STEGANOGRAPHY METRICS:")
        lsb_2bit_metrics = calculate_image_quality_metrics(cover_array, stego_2bit)
        lsb_4bit_metrics = calculate_image_quality_metrics(cover_array, stego_4bit)
        
        print(f"2-bit LSB - PSNR: {lsb_2bit_metrics['psnr']:.2f} dB, SSIM: {lsb_2bit_metrics['ssim']:.4f}")
        print(f"4-bit LSB - PSNR: {lsb_4bit_metrics['psnr']:.2f} dB, SSIM: {lsb_4bit_metrics['ssim']:.4f}")
        
        print("\nDCT STEGANOGRAPHY METRICS:")
        dct_low_metrics = calculate_image_quality_metrics(cover_array, stego_dct_low)
        dct_high_metrics = calculate_image_quality_metrics(cover_array, stego_dct_high)
        
        print(f"DCT α=0.05 - PSNR: {dct_low_metrics['psnr']:.2f} dB, SSIM: {dct_low_metrics['ssim']:.4f}")
        print(f"DCT α=0.15 - PSNR: {dct_high_metrics['psnr']:.2f} dB, SSIM: {dct_high_metrics['ssim']:.4f}")
        
        # Secret image recovery quality
        print("\nSECRET IMAGE RECOVERY QUALITY:")
        secret_array = np.array(secret_img.resize(cover_img.size))
        
        lsb_2bit_recovery = calculate_image_quality_metrics(secret_array, extracted_2bit)
        lsb_4bit_recovery = calculate_image_quality_metrics(secret_array, extracted_4bit)
        dct_low_recovery = calculate_image_quality_metrics(secret_array, extracted_dct_low)
        dct_high_recovery = calculate_image_quality_metrics(secret_array, extracted_dct_high)
        
        print(f"LSB 2-bit recovery - PSNR: {lsb_2bit_recovery['psnr']:.2f} dB, SSIM: {lsb_2bit_recovery['ssim']:.4f}")
        print(f"LSB 4-bit recovery - PSNR: {lsb_4bit_recovery['psnr']:.2f} dB, SSIM: {lsb_4bit_recovery['ssim']:.4f}")
        print(f"DCT α=0.05 recovery - PSNR: {dct_low_recovery['psnr']:.2f} dB, SSIM: {dct_low_recovery['ssim']:.4f}")
        print(f"DCT α=0.15 recovery - PSNR: {dct_high_recovery['psnr']:.2f} dB, SSIM: {dct_high_recovery['ssim']:.4f}")
        
        # Method comparison
        print("\n" + "="*60)
        print("METHOD COMPARISON & ANALYSIS")
        print("="*60)
        
        print("\nLSB STEGANOGRAPHY:")
        print("ADVANTAGES:")
        print("  + Simple implementation")
        print("  + High embedding capacity")
        print("  + Perfect recovery (lossless)")
        print("  + Fast processing")
        print("DISADVANTAGES:")
        print("  - Vulnerable to statistical analysis")
        print("  - Easily detected by steganalysis tools")
        print("  - Fragile to image processing operations")
        print("  - Visible artifacts in high embedding rates")
        
        print("\nDCT STEGANOGRAPHY:")
        print("ADVANTAGES:")
        print("  + More robust to image processing")
        print("  + Better resistance to steganalysis")
        print("  + Perceptually invisible at low embedding rates")
        print("  + Works well with JPEG compression")
        print("DISADVANTAGES:")
        print("  - Lower embedding capacity")
        print("  - More complex implementation")
        print("  - Lossy recovery of secret data")
        print("  - Requires original cover for extraction")
        
        print("\nSECURITY CONSIDERATIONS:")
        print("1. DETECTION RESISTANCE:")
        print(f"   - LSB methods: Vulnerable to chi-square and histogram analysis")
        print(f"   - DCT methods: Better resistance to statistical attacks")
        
        print("\n2. ROBUSTNESS:")
        print(f"   - LSB: Fragile to any image modifications")
        print(f"   - DCT: Survives mild compression and filtering")
        
        print("\n3. CAPACITY vs STEALTH TRADE-OFF:")
        print(f"   - LSB can hide more data but less securely")
        print(f"   - DCT hides less data but more securely")
        
        print("\n4. APPLICATIONS:")
        print("   LSB APPLICATIONS:")
        print("   - Quick data hiding in controlled environments")
        print("   - High-capacity covert communication")
        print("   - Digital watermarking (when robustness not critical)")
        
        print("   DCT APPLICATIONS:")
        print("   - Robust digital watermarking")
        print("   - Copyright protection")
        print("   - Covert communication in compressed media")
        print("   - Anti-forensic techniques")
        
        # Create quality comparison chart
        print("\nGenerating quality comparison chart...")
        
        methods = ['LSB 2-bit', 'LSB 4-bit', 'DCT α=0.05', 'DCT α=0.15']
        cover_psnr = [lsb_2bit_metrics['psnr'], lsb_4bit_metrics['psnr'], 
                      dct_low_metrics['psnr'], dct_high_metrics['psnr']]
        recovery_psnr = [lsb_2bit_recovery['psnr'], lsb_4bit_recovery['psnr'],
                        dct_low_recovery['psnr'], dct_high_recovery['psnr']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cover image quality
        bars1 = ax1.bar(methods, cover_psnr, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
        ax1.set_title('Cover Image Quality (PSNR)', fontweight='bold')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_ylim(0, max(cover_psnr) * 1.1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, cover_psnr):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Secret recovery quality
        bars2 = ax2.bar(methods, recovery_psnr, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
        ax2.set_title('Secret Recovery Quality (PSNR)', fontweight='bold')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_ylim(0, max(recovery_psnr) * 1.1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, recovery_psnr):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.suptitle('Steganography Methods: Quality Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save quality comparison
        quality_filename = 'Lecture_9_CCD_Methods/steganography_quality_comparison.png'
        plt.savefig(quality_filename, dpi=300, bbox_inches='tight')
        print(f"Quality comparison saved as: {quality_filename}")
        
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: Could not find image files.")
        print(f"Please ensure the following files exist:")
        print(f"- Lecture_9_CCD_Methods/sample_images/cover.jpg")
        print(f"- Lecture_9_CCD_Methods/sample_images/secret.jpg")
        print(f"\nError details: {e}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. LSB methods offer high capacity but low security")
    print("2. DCT methods provide better security but lower capacity")
    print("3. Choice of method depends on specific requirements:")
    print("   - Use LSB for high-capacity, low-security applications")
    print("   - Use DCT for robust, secure steganography")
    print("4. Both methods demonstrate the concealment aspect of CCD")

if __name__ == "__main__":
    main()
