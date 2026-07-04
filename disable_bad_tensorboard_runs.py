import os
from tensorboard.backend.event_processing import event_accumulator

LOGDIR = "models/raw_models/DepthAnythingV2-revised/checkpoints/basic_finetuning"
THRESHOLD = 1.0
SCALAR_TAG = "train/loss"

for root, dirs, files in os.walk(LOGDIR):
    for file in files:
        if "tfevents" in file:
            event_path = os.path.join(root, file)

            try:
                ea = event_accumulator.EventAccumulator(event_path)
                ea.Reload()

                if SCALAR_TAG not in ea.Tags()["scalars"]:
                    continue

                events = ea.Scalars(SCALAR_TAG)
                if not events:
                    continue

                final_loss = events[-1].value
                print(f"{event_path} -> final loss: {final_loss}")

                if final_loss > THRESHOLD:
                    new_filename = file.replace("tfevents", "DISABLED")
                    new_path = os.path.join(root, new_filename)
                    os.rename(event_path, new_path)
                    print(f"Renamed to: {new_path}")

            except Exception as e:
                print(f"Error reading {event_path}: {e}")
