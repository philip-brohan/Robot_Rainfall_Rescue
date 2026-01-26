# Custom trainer
# With callbacks for saving the model and reporting losse

import os
from transformers import TrainerCallback
from trl import SFTTrainer


# Modify the trainer class to add a reference to itself in the global namespace for callback access
# There's got to be a better way of making the trainer available to callbacks...
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # keep a global reference for callbacks that expect globals().get("trainer")
        globals()["trainer"] = self


# Define a callback to save a merged version of the model at the end of each epoch
class SaveStateCallback(TrainerCallback):
    def __init__(self, out_dir):
        self.out_dir = out_dir

    # Called at end of epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        print("[SaveStateCallback] on_epoch_end called")
        tr = globals().get("trainer")

        save_dir = os.path.join(
            self.out_dir, f"merged_epoch_{int(getattr(state,'epoch',0))}"
        )
        os.makedirs(save_dir, exist_ok=True)
        tr.save_model(output_dir=save_dir)


# Define a callback to give the training set loss and evaluation set loss at the end of each epoch
class CompareLossCallback(TrainerCallback):
    def __init__(self, base_model_id, out_dir):
        self.base_model_id = base_model_id
        self.out_dir = out_dir

    # Called at end of epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        print("[CompareLossCallback] on_epoch_end called")
        tr = globals().get("trainer")
        train_loss = tr.evaluate(eval_dataset=getattr(tr, "train_dataset", None))
        print("Training loss:", train_loss)
        eval_loss = tr.evaluate(eval_dataset=getattr(tr, "eval_dataset", None))
        print("Test loss:", eval_loss)
