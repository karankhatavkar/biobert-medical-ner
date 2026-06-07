"""Device selection helper with Apple Silicon (MPS) support."""
import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA > Apple Silicon MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    dev = get_device()
    print(f"Selected device: {dev}")
    if dev.type == "mps":
        print("Apple Silicon GPU (Metal) acceleration is active.")
