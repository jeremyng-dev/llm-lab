import torch
import tiktoken
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def countParams(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def textToTokens(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def tokensToText(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    return tokenizer.decode(token_ids.squeeze().tolist())


def generateText(
    model: nn.Module,
    prompt: str,
    context_length: int,
    device: torch.device,
    tokenizer: tiktoken.Encoding,
    max_length: int = 50,
    temperature: float = 0.0,
    top_k: int = None,
    eos_id: int = None,
) -> str:
    ids = textToTokens(prompt, tokenizer)
    ids = ids.to(device)
    for _ in range(max_length):
        ids_cut = ids[:, -context_length:]
        with torch.no_grad():
            logits = model(ids_cut)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k = top_k_values[:, -1]
            logits = torch.where(
                logits < min_top_k, torch.tensor(float("-inf")), logits
            )

        if temperature > 0.0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and next_token_id == eos_id:
            break

        ids = torch.cat([ids, next_token_id], dim=1)

    return tokensToText(ids, tokenizer)
