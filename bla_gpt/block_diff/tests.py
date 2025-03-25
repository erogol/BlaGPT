import unittest
import torch
import sys
import os

# Import the model
from model import BD3LM, RotaryPositionalEmbedding


class TestRotaryPositionalEmbedding(unittest.TestCase):
    """Tests for the RotaryPositionalEmbedding class."""
    
    def setUp(self):
        self.dim = 64
        self.max_seq_len = 128
        self.rope = RotaryPositionalEmbedding(self.dim, self.max_seq_len)
        
    def test_forward_shape(self):
        """Test that the forward pass produces the correct shape."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, self.dim)
        output = self.rope(x)
        self.assertEqual(output.shape, (batch_size, seq_len, self.dim))
    
    def test_rotation(self):
        """Test that the rotation property is maintained."""
        # Create a simple input with known values to test rotation
        x = torch.zeros(1, 2, self.dim)
        
        # Set specific values in dimensions that will be rotated
        # Use pairs to maintain norm after rotation
        x[0, 0, 0] = 1.0
        x[0, 0, 1] = 0.0
        x[0, 1, 0] = 0.0
        x[0, 1, 1] = 1.0
        
        # Calculate original norms
        orig_norm_0 = torch.norm(x[0, 0]).item()
        orig_norm_1 = torch.norm(x[0, 1]).item()
        
        # Apply rotary embeddings
        output = self.rope(x)
        
        # Verify rotation has occurred (values have changed)
        rotated = False
        if (abs(output[0, 0, 0].item() - 1.0) > 1e-5 or 
            abs(output[0, 0, 1].item() - 0.0) > 1e-5 or
            abs(output[0, 1, 0].item() - 0.0) > 1e-5 or
            abs(output[0, 1, 1].item() - 1.0) > 1e-5):
            rotated = True
        
        self.assertTrue(rotated, "Rotary embeddings should change input values")
        
        # Check norm preservation with more relaxed tolerance
        self.assertAlmostEqual(
            orig_norm_0,
            torch.norm(output[0, 0]).item(),
            places=2,
            msg="Norm should be preserved for token at position 0"
        )
        self.assertAlmostEqual(
            orig_norm_1,
            torch.norm(output[0, 1]).item(),
            places=2,
            msg="Norm should be preserved for token at position 1"
        )


class TestBD3LM(unittest.TestCase):
    """Tests for the BD3LM class."""
    
    def setUp(self):
        self.vocab_size = 1000
        self.mask_id = 999  # Last token is mask
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 2
        self.block_size = 4
        self.max_seq_len = 16
        
        self.model = BD3LM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=128,  # Smaller for testing
            dropout=0.1,
            max_seq_len=self.max_seq_len,
            block_size=self.block_size,
            mask_id=self.mask_id
        )
    
    def test_model_initialization(self):
        """Test that the model was initialized correctly."""
        self.assertEqual(self.model.vocab_size, self.vocab_size)
        self.assertEqual(self.model.d_model, self.d_model)
        self.assertEqual(self.model.block_size, self.block_size)
        self.assertEqual(self.model.mask_id, self.mask_id)
        self.assertEqual(len(self.model.layers), self.num_layers)
        
    def test_block_causal_mask(self):
        """Test that the block-causal attention mask is created correctly."""
        seq_len = 8
        block_size = 4
        device = torch.device("cpu")
        
        mask = self.model._create_block_causal_mask(seq_len, block_size, device)
        
        # Check shape
        self.assertEqual(mask.shape, (seq_len, seq_len))
        
        # Check that tokens can attend to tokens in the same block
        self.assertEqual(mask[0, 0].item(), 0)  # Can attend (0 means attend, -inf means don't)
        self.assertEqual(mask[3, 0].item(), 0)  # Same block
        
        # Check that tokens cannot attend to future blocks
        self.assertLess(mask[0, 4].item(), -1e8)  # Cannot attend (very negative means don't attend)
        
        # Check that tokens can attend to previous blocks
        self.assertEqual(mask[4, 0].item(), 0)  # Can attend to previous block
    
    def test_forward_pass(self):
        """Test that the forward pass runs without errors and returns expected shapes."""
        batch_size = 2
        seq_len = 8
        
        # Create random input IDs and target IDs
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        
        # Make some input tokens masked
        input_ids[0, 2] = self.mask_id
        input_ids[1, 5] = self.mask_id
        
        # Set noise level
        noise_level = torch.tensor([0.5, 0.7])
        
        # Run forward pass
        output = self.model(input_ids, target_ids, noise_level)
        
        # Check that loss and logits are returned
        self.assertIn('loss', output)
        self.assertIn('logits', output)
        
        # Check shapes
        self.assertEqual(output['logits'].shape, (batch_size, seq_len, self.vocab_size))
        self.assertEqual(output['loss'].shape, ())  # Scalar loss
    
    def test_loss_calculation(self):
        """Test that the BD3 loss is calculated correctly."""
        batch_size = 2
        seq_len = 8
        
        # Create logits (predictions for each token)
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        
        # Create input IDs with some masked tokens
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        input_ids[0, 2] = self.mask_id
        input_ids[1, 5] = self.mask_id
        
        # Create target IDs (the true tokens)
        target_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        
        # Set noise level
        noise_level = torch.tensor([0.5, 0.7])
        
        # Calculate loss
        loss = self.model._calculate_bd3_loss(logits, input_ids, target_ids, noise_level)
        
        # Loss should be scalar
        self.assertEqual(loss.shape, ())
        
        # Loss should be positive
        self.assertGreater(loss.item(), 0)
    
    def test_generate(self):
        """Test that the generate method produces sequences of the expected shape."""
        batch_size = 2
        prompt_len = 4
        max_len = 12
        
        # Create a prompt
        prompt = torch.randint(0, self.vocab_size, (batch_size, prompt_len))
        
        # Generate sequence
        with torch.no_grad():
            generated = self.model.generate(
                prompt=prompt,
                max_len=max_len,
                num_diffusion_steps=10,  # Use fewer steps for testing
                temperature=1.0,
                top_p=0.9
            )
        
        # Check shape
        self.assertEqual(generated.shape[0], batch_size)
        self.assertLessEqual(generated.shape[1], max_len)
        
        # Check that prompt is preserved
        torch.testing.assert_close(generated[:, :prompt_len], prompt)
        
        # Check that no mask tokens remain in the generated sequence
        self.assertFalse(torch.any(generated == self.mask_id))


if __name__ == "__main__":
    unittest.main()