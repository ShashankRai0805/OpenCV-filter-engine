import torch

print("✅ PyTorch version:", torch.__version__)
print("✅ Python executable:", torch.__file__)

# Check CUDA / GPU availability
if torch.cuda.is_available():
    print("⚡ CUDA is available!")
    print("🔹 GPU Name:", torch.cuda.get_device_name(0))
else:
    print("💻 Running on CPU (no GPU detected).")

# Simple tensor test
x = torch.rand(3, 3)
print("\nRandom tensor:\n", x)
print("\n✅ PyTorch is working correctly!")
    