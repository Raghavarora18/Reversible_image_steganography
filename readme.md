This project implements a reversible image steganography system that securely hides a secret image inside a cover image using Discrete Wavelet Transform (DWT) and deep learning-based feature extraction. The hidden image can be recovered without loss, ensuring high security and imperceptibility.

The system focuses on maintaining high visual quality, achieving strong PSNR and SSIM values while preventing visible distortion. A Streamlit-based user interface is provided for easy image embedding and extraction.

Objectives:
To securely embed a secret image inside a cover image
To ensure lossless recovery of the secret image
To minimize visual distortion in the stego image
To provide a simple and interactive UI for demonstration

Tech Stack:
Programming Language: Python
Deep Learning: PyTorch
Image Processing: OpenCV, NumPy
UI Framework: Streamlit
Transform: Discrete Wavelet Transform (DWT)
Version Control: Git & GitHub

Evaluation Metrics:
PSNR (Peak Signal-to-Noise Ratio) – Measures image quality
SSIM (Structural Similarity Index) – Measures structural similarity
MSE (Mean Squared Error) – Measures reconstruction error
High values of PSNR and SSIM indicate effective steganography with minimal distortion.