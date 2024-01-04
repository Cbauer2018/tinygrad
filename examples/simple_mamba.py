from extra.models.mamba import Mamba, generate
from transformers import AutoTokenizer


model = Mamba.from_pretrained('state-spaces/mamba-790m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
generate(model, tokenizer, 'Mamba is')