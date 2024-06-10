from trainer.base import Trainer, MultiModalTrainer

def make_trainer(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    **kwargs
):
    return Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        **kwargs
    )


def make_multimodal_trainer(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    **kwargs
):
    return MultiModalTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        **kwargs
    )