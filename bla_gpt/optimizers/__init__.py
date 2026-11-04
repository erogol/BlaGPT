import importlib
from typing import List

import torch


def describe_update_paths(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    name_keys: tuple = ("wte", "embed", "lm_head"),
) -> List[str]:
    """Runtime probe: which update rule touched each embedding-like tensor.

    Call after the first optimizer.step(). The rule is read from the
    optimizer state that the step created: ``sgd_momentum_buffer`` (F102
    SGD-M group), ``momentum_buffer`` (Aurora), ``moment1`` (AdamW backup).
    """
    head = getattr(getattr(model, "lm_head", None), "weight", None)
    lines = []
    for name, p in model.named_parameters():
        if not any(key in name for key in name_keys):
            continue
        state = optimizer.state.get(p, {})
        if "sgd_momentum_buffer" in state:
            path = "sgd_m"
        elif "momentum_buffer" in state:
            path = "aurora"
        elif "moment1" in state:
            path = "adamw"
        else:
            path = "none"
        lines.append(
            f"[update_path_probe step=1] param={name} shape={tuple(p.shape)} "
            f"tied_lm_head={p is head} path={path} state_keys={sorted(state.keys())}"
        )
    return lines


def get_optimizer(
    optimizer_name: str,
    optimizer_params: dict,
    lr: float,
    model: torch.nn.Module = None,
    parameters: List = None,
) -> torch.optim.Optimizer:
    """Find, initialize and return a Torch optimizer.

    Args:
        optimizer_name (str): Optimizer name.
        optimizer_params (dict): Optimizer parameters.
        lr (float): Initial learning rate.
        model (torch.nn.Module): Model to pass to the optimizer.

    Returns:
        torch.optim.Optimizer: Functional optimizer.
    """
    if model is not None:
        parameters = model.parameters()

    if optimizer_name.lower() == "radam":
        module = importlib.import_module("optimizers.radam")
        optimizer = getattr(module, "RAdam")
    elif optimizer_name.lower() == "palm_soap":
        try:
            from heavyball import PaLMForeachSOAP
        except ImportError:
            raise ImportError(
                "To use PaLMForeachSOAP, please install the heavyball package."
            )

        module = PaLMForeachSOAP
    elif optimizer_name.lower() == "ademamix":
        module = importlib.import_module("optimizers.ademamix")
        optimizer = getattr(module, "AdEMAMix")
    elif optimizer_name.lower() == "adopt":
        module = importlib.import_module("optimizers.adopt")
        optimizer = getattr(module, "ADOPT")
        return optimizer(parameters, lr=lr, **optimizer_params, decoupled=True)
    elif optimizer_name.lower() == "adamw_indep":
        module = importlib.import_module("optimizers.adamw_indep_weight_decay")
        optimizer = getattr(module, "AdamW")
    elif optimizer_name.lower() == "c_adamw":
        module = importlib.import_module("optimizers.c_adamw")
        optimizer = getattr(module, "AdamW")
    elif optimizer_name.lower() == "demo":
        module = importlib.import_module("optimizers.demo")
        optimizer = getattr(module, "DeMo")
    elif optimizer_name.lower() == "adam_mini":
        try:
            from adam_mini import Adam_mini
        except ImportError:
            raise ImportError("To use Adam_mini, please install the adam-mini package.")

        optimizer = Adam_mini(
            named_parameters=model.named_parameters(), lr=lr, **optimizer_params
        )
        optimizer.wqk_names.add("kv_proj")
        optimizer.attn_proj_names.add("c_proj")

        return optimizer
    elif optimizer_name.lower() == "mars":
        module = importlib.import_module("optimizers.mars")
        optimizer = getattr(module, "MARS")
    elif optimizer_name.lower() == "muon":
        module = importlib.import_module("optimizers.muon")
        optimizer = getattr(module, "Muon")
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim < 2 or "embed_tokens" in name or "lm_head"  in name

        ]
        # Extract CWD flag if present
        use_cwd = optimizer_params.pop('use_cautious_weight_decay', False)

        return optimizer(
            lr=lr,
            muon_params=muon_params,
            adamw_params=adamw_params,
            use_cautious_weight_decay=use_cwd,
            **optimizer_params
        )
    elif optimizer_name.lower() == "adamuon":
        module = importlib.import_module("optimizers.adamuon")
        optimizer = getattr(module, "AdaMuon")
        hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2]
        non_hidden_params = [
            p
            for name, p in model.named_parameters()
            if  "embed_tokens" in name and "lm_head"  in name
        ]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        return optimizer(param_groups, lr=lr, **optimizer_params)
    elif optimizer_name.lower() == "normuon":
        module = importlib.import_module("optimizers.normuon")
        optimizer = getattr(module, "HybridNorMuon")
        normuon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim < 2 or "embed_tokens" in name or "lm_head" in name
        ]
        return optimizer(
            lr=lr,
            normuon_params=normuon_params,
            adamw_params=adamw_params,
            **optimizer_params
        )

    elif optimizer_name.lower() == "aurora":
        module = importlib.import_module("optimizers.aurora")
        optimizer = getattr(module, "Aurora")
        # F102 (arXiv 2610.38277): optional SGD-M routing for the tied
        # embedding / LM head tensor. Default off = bit-identical Aurora.
        lm_head_sgd_m = optimizer_params.pop("lm_head_sgd_m", False)
        lm_head_sgd_lr = optimizer_params.pop("lm_head_sgd_lr", 0.01)
        aurora_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim < 2 or "embed_tokens" in name or "lm_head" in name
        ]
        sgd_m_params = []
        # The tied tensor is the lm_head weight object. With weight tying the
        # input embedding (e.g. transformer.wte.weight) shares this object, so
        # named_parameters() yields it once under the embedding's name.
        head_tensor = getattr(getattr(model, "lm_head", None), "weight", None)
        if lm_head_sgd_m:
            if head_tensor is None:
                raise ValueError(
                    "lm_head_sgd_m=True requires model.lm_head.weight (the tied head tensor)"
                )
            sgd_m_params = [head_tensor]
            aurora_params = [p for p in aurora_params if p is not head_tensor]
            adamw_params = [p for p in adamw_params if p is not head_tensor]
        # Routing probe (always on, before step 1): report which update rule
        # holds the tied embedding / LM head tensor. Lands in run.log.
        if head_tensor is not None:
            for name, p in model.named_parameters():
                if p is not head_tensor:
                    continue
                if any(p is q for q in sgd_m_params):
                    rule = f"sgd_m(lr={lm_head_sgd_lr}, momentum=0.99)"
                elif any(p is q for q in aurora_params):
                    rule = "aurora"
                else:
                    rule = "adamw"
                print(
                    f"[lm_head_probe] tied lm_head/embedding tensor param_name={name} "
                    f"shape={tuple(head_tensor.shape)} update_rule={rule}"
                )
                break
        return optimizer(
            lr=lr,
            aurora_params=aurora_params,
            adamw_params=adamw_params,
            sgd_m_params=sgd_m_params,
            sgd_m_lr=lm_head_sgd_lr,
            **optimizer_params
        )
    elif optimizer_name.lower() == "biclip":
        from optimizers.bliclip import BiClipSGD_Full
        return BiClipSGD_Full(parameters, lr=lr, **optimizer_params)
    else:
        optimizer = getattr(torch.optim, optimizer_name)
    return optimizer(parameters, lr=lr, **optimizer_params)
