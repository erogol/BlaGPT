import torch
import argparse
from model import BD3LM
from utils import apply_noise


def main():
    parser = argparse.ArgumentParser(description="Run a simple example of the BD3LM model")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension (smaller for example)")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--block_size", type=int, default=4, help="Block size for block diffusion")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to run on")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Initialize model with smaller size for example
    model = BD3LM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=64,
        block_size=args.block_size,
        mask_id=args.vocab_size - 1,  # Assuming the last token is [MASK]
    ).to(device)
    
    # Create a simple sequence for demonstration
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, args.vocab_size - 1, (batch_size, seq_len), device=device)
    
    print("Original sequence shape:", input_ids.shape)
    
    # Apply noise to create masked input
    noise_level = torch.tensor([0.3, 0.7], device=device)  # Different noise levels for each batch item
    masked_input = apply_noise(input_ids, model.mask_id, noise_level)
    
    print("Masked sequence (30% and 70% mask rate):")
    print(masked_input)
    
    # Forward pass
    output = model(masked_input, input_ids, noise_level)
    
    print("Loss:", output['loss'].item())
    print("Logits shape:", output['logits'].shape)
    
    # Generate a sequence
    print("\nGenerating a sequence of length 20...")
    prompt = torch.randint(0, args.vocab_size - 1, (1, 4), device=device)
    print("Prompt:", prompt)
    
    with torch.no_grad():
        generated = model.generate(
            prompt=prompt,
            max_len=20,
            num_diffusion_steps=20,
            temperature=1.0,
            top_p=0.9,
            device=device
        )
    
    print("Generated sequence shape:", generated.shape)
    print("Generated sequence:", generated)


if __name__ == "__main__":
    main()