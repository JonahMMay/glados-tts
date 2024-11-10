import torch
from .text.cleaners import Cleaner
from .text.tokenizer import Tokenizer

def prepare_text(text: str, models_dir: str) -> torch.Tensor:
    if not text[-1] in '.?!':
        text += '.'
    cleaner = Cleaner('english_cleaners', True, 'en-us', models_dir=models_dir)
    tokenizer = Tokenizer()
    tokens = tokenizer(cleaner(text))
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
