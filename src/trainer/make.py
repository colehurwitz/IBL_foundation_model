from trainer.base import Trainer

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