import types

import pytest

from ai_assistant_pro import engine as engine_pkg


class DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self):
        self.encoded = []

    def encode(self, text, return_tensors=None):
        self.encoded.append(text)

        class _Fake:
            shape = (1, 1)

            def to(self, device):
                return self

        return _Fake()

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class DummyModel:
    def __init__(self):
        self.config = types.SimpleNamespace(
            num_attention_heads=1,
            hidden_size=1,
            num_hidden_layers=1,
        )

    def parameters(self):
        return [types.SimpleNamespace(numel=lambda: 1)]

    def eval(self):
        return self

    def generate(self, *args, **kwargs):
        return [[0, 1]]


@pytest.fixture(autouse=True)
def patch_hf(monkeypatch):
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    # Patch inside engine.model import path
    monkeypatch.setitem(
        engine_pkg.__dict__,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=lambda *_, **__: dummy_model),
    )
    monkeypatch.setitem(
        engine_pkg.__dict__,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *_, **__: dummy_tokenizer),
    )
    monkeypatch.setitem(engine_pkg.__dict__, "AutoConfig", types.SimpleNamespace)
    # Prevent transformers hub lookups entirely
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Also patch the module import path used in engine.model
    import ai_assistant_pro.engine.model as model_mod
    monkeypatch.setattr(model_mod, "AutoModelForCausalLM", engine_pkg.AutoModelForCausalLM)
    monkeypatch.setattr(model_mod, "AutoTokenizer", engine_pkg.AutoTokenizer)
    monkeypatch.setattr(model_mod, "AutoConfig", types.SimpleNamespace)
    yield


def test_engine_generate_uses_tokenizer_and_model(monkeypatch):
    from ai_assistant_pro.engine.model import AssistantEngine

    engine = AssistantEngine(model_name="stub-model", use_triton=False, enable_paged_attention=False)
    text = engine.generate("hi", max_tokens=2)

    assert text == "decoded"
    assert engine.tokenizer.pad_token == engine.tokenizer.eos_token


def test_engine_counts_parameters(monkeypatch):
    from ai_assistant_pro.engine.model import AssistantEngine

    engine = AssistantEngine(model_name="stub-model", use_triton=False, enable_paged_attention=False)
    assert engine._count_parameters() == 1
