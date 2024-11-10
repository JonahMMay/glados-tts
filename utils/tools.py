import torch
from pathlib import Path
from typing import Optional
from .text.cleaners import Cleaner
from .text.tokenizer import Tokenizer

def prepare_text(
    text: str,
    models_dir: Path,
    device: torch.device,
    cleaner_name: str = 'english_cleaners',
    use_phonemes: bool = True,
    lang: str = 'en-us'
) -> torch.Tensor:
    """
    Prepares text for input into the TTS model by cleaning, tokenizing, and converting to a tensor.

    Args:
        text: The input text to process.
        models_dir: Path to the directory containing model files.
        device: The torch device to use (e.g., torch.device('cpu') or torch.device('cuda')).
        cleaner_name: Name of the cleaning function to use.
        use_phonemes: Whether to convert text to phonemes.
        lang: Language code for phonemization.

    Returns:
        A torch.Tensor containing the tokenized text ready for the model.
    """
    if not text:
        raise ValueError("Input text cannot be empty.")

    if not text[-1] in '.?!':
        text += '.'

    cleaner = Cleaner(
        cleaner_name=cleaner_name,
        use_phonemes=use_phonemes,
        lang=lang,
        models_dir=models_dir,
        device=str(device)  # Assuming Cleaner expects device as a string
    )
    tokenizer = Tokenizer()
    cleaned_text = cleaner(text)
    tokens = tokenizer(cleaned_text)

    return torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
