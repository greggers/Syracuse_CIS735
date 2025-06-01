
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

def main():
    """
    Main function to demonstrate steganography techniques
    """
    print("="*60)
    print("STEGANOGRAPHY DEMONSTRATION")
    print("Concealment, Camouflage, and Deception (CCD) Methods")
    print("="*60)
    
    print("\nSteganography: The art of hiding information in plain sight")
    print("This demonstration shows how to conceal a secret image within a cover image")
    print("using least significant bit (LSB) manipulation.\n")
    
    try:
        # Load images
        print("Loading images...")
        cover_img = Image.open('Lecture_9_CCD_Methods/sample_images/cover.jpg').convert('L')  # Convert to grayscale
        secret_img = Image.open('Lecture_9_CCD_Methods/sample_images/secret.jpg').convert('L')
        
        print(f"Cover image size: {cover_img.size}")
        print(f"Secret image size: {secret_img.size}")
        
        # Create two versions with different bit depths
        print("\nCreating steganographic images...")
        print("- 2-bit version: Less visible distortion, some quality loss in hidden image")
        print("- 4-bit version: More visible distortion, better quality in hidden image")
        
        stego_2bit = hide_image(cover_img, secret_img, bits_to_use=2)  # Less visible, some aliasing
        stego_4bit = hide_image(cover_img, secret_img, bits_to_use=4)  # More visible, more aliasing
        
        # Extract hidden images
        print("\nExtracting hidden images...")
        extracted_2bit = extract_image(stego_2bit, bits_used=2)
        extracted_4bit = extract_image(stego_4bit, bits_used=4)
        
        # Display results
        print("\nGenerating visualization...")
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.imshow(cover_img, cmap='gray')
        plt.title('Original Cover Image')
        plt.axis('off')
        
        plt.subplot(232)
        plt.imshow(secret_img, cmap='gray')
        plt.title('Secret Image')
        plt.axis('off')
        
        plt.subplot(233)
        plt.imshow(stego_2bit, cmap='gray')
        plt.title('Steganography (2 bits)\nSubtle concealment')
        plt.axis('off')
        
        plt.subplot(234)
        plt.imshow(stego_4bit, cmap='gray')
        plt.title('Steganography (4 bits)\nMore obvious distortion')
        plt.axis('off')
        
        plt.subplot(235)
        plt.imshow(extracted_2bit, cmap='gray')
        plt.title('Extracted (2 bits)\nLower quality recovery')
        plt.axis('off')
        
        plt.subplot(236)
        plt.imshow(extracted_4bit, cmap='gray')
        plt.title('Extracted (4 bits)\nHigher quality recovery')
        plt.axis('off')
        
        plt.suptitle('Steganography: Concealment & Deception Demonstration', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure
        output_filename = 'Lecture_9_CCD_Methods/steganography_demonstration.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as: {output_filename}")
        
        plt.show()
        
        # Demonstrate concealment effectiveness
        print("\n" + "="*50)
        print("CONCEALMENT ANALYSIS")
        print("="*50)
        
        # Calculate differences
        cover_array = np.array(cover_img)
        stego_2bit_diff = np.mean(np.abs(cover_array.astype(float) - stego_2bit.astype(float)))
        stego_4bit_diff = np.mean(np.abs(cover_array.astype(float) - stego_4bit.astype(float)))
        
        print(f"Average pixel difference (2-bit): {stego_2bit_diff:.2f}")
        print(f"Average pixel difference (4-bit): {stego_4bit_diff:.2f}")
        
        # Calculate signal-to-noise ratio
        def calculate_psnr(original, modified):
            mse = np.mean((original.astype(float) - modified.astype(float)) ** 2)
            if mse == 0:
                return float('inf')
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            return psnr
        
        psnr_2bit = calculate_psnr(cover_array, stego_2bit)
        psnr_4bit = calculate_psnr(cover_array, stego_4bit)
        
        print(f"\nPeak Signal-to-Noise Ratio (PSNR):")
        print(f"2-bit steganography: {psnr_2bit:.2f} dB")
        print(f"4-bit steganography: {psnr_4bit:.2f} dB")
        print("(Higher PSNR = better concealment)")
        
        # Security analysis
        print(f"\n" + "="*50)
        print("SECURITY CONSIDERATIONS")
        print("="*50)
        print("1. DETECTION RESISTANCE:")
        print("   - 2-bit method: High resistance to visual detection")
        print("   - 4-bit method: Moderate resistance, may be detectable")
        
        print("\n2. CAPACITY vs STEALTH TRADE-OFF:")
        print("   - More bits = better secret image quality")
        print("   - Fewer bits = better concealment")
        
        print("\n3. POTENTIAL COUNTERMEASURES:")
        print("   - Statistical analysis of LSBs")
        print("   - Histogram analysis")
        print("   - Chi-square attacks")
        print("   - Visual inspection")
        
        print("\n4. APPLICATIONS:")
        print("   - Covert communication")
        print("   - Digital watermarking")
        print("   - Copyright protection")
        print("   - Data exfiltration (malicious use)")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find image files.")
        print(f"Please ensure the following files exist:")
        print(f"- Lecture_9_CCD_Methods/sample_images/cover.jpg")
        print(f"- Lecture_9_CCD_Methods/sample_images/secret.jpg")
        print(f"\nError details: {e}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    print(f"\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
