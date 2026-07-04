import json
from datetime import datetime
from pathlib import Path


class TrainingStateManager:
    def __init__(self, storage_dir):
        self.storage_dir = Path(storage_dir)
        self.state_file = self.storage_dir / "training_state.json"
        self.progress_file = self.storage_dir / "training_progress.json"
        self.stop_flag_file = self.storage_dir / "training_stop_flag"

        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_training_state(self, process_id, start_time):
        state = {"process_id": process_id, "start_time": start_time.isoformat()}
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def load_training_state(self):
        if not self.state_file.exists():
            return None
        with open(self.state_file, "r") as f:
            state = json.load(f)
        state["start_time"] = datetime.fromisoformat(state["start_time"])
        return state

    def clear_training_state(self):
        for file in [self.state_file, self.progress_file, self.stop_flag_file]:
            if file.exists():
                file.unlink()

    def save_progress(self, progress, current_epoch):
        data = {"progress": progress, "current_epoch": current_epoch}
        with open(self.progress_file, "w") as f:
            json.dump(data, f)

    def load_progress(self):
        if not self.progress_file.exists():
            return {"progress": 0, "current_epoch": 0}
        with open(self.progress_file, "r") as f:
            data = json.load(f)
        return data

    def set_stop_flag(self):
        self.stop_flag_file.touch()

    def clear_stop_flag(self):
        if self.stop_flag_file.exists():
            self.stop_flag_file.unlink()

    def check_stop_flag(self):
        return self.stop_flag_file.exists()

    def save_loss_history(self, loss_history: list) -> None:
        with open(self.storage_dir / "loss_history.json", "w") as f:
            json.dump(loss_history, f)

    def load_loss_history(self) -> list:
        loss_file = self.storage_dir / "loss_history.json"
        if not loss_file.exists():
            return []
        with open(loss_file) as f:
            return json.load(f)

    def clear_loss_history(self) -> None:
        loss_file = self.storage_dir / "loss_history.json"
        if loss_file.exists():
            loss_file.unlink()

    def append_history_entry(self, entry: dict) -> None:
        history = self.load_history()
        history.insert(0, entry)  # newest first
        with open(self.storage_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    def load_history(self) -> list:
        history_file = self.storage_dir / "training_history.json"
        if not history_file.exists():
            return []
        with open(history_file) as f:
            return json.load(f)
