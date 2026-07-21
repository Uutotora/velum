import json
from datetime import datetime
from pathlib import Path


class PredictionStateManager:
    def __init__(self, storage_dir):
        self.storage_dir = Path(storage_dir)
        self.state_file = self.storage_dir / "prediction_state.json"
        self.progress_file = self.storage_dir / "prediction_progress.json"
        self.stop_flag_file = self.storage_dir / "prediction_stop_flag"

        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_prediction_state(self, process_id, start_time, total_images):
        state = {"process_id": process_id, "start_time": start_time.isoformat(), "total_images": total_images}
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def load_prediction_state(self):
        if not self.state_file.exists():
            return None
        with open(self.state_file, "r") as f:
            state = json.load(f)
        state["start_time"] = datetime.fromisoformat(state["start_time"])
        return state

    def clear_prediction_state(self):
        for file in [self.state_file, self.progress_file, self.stop_flag_file]:
            if file.exists():
                file.unlink()

    def save_progress(self, done: int, total: int = None,
                      current_image: str = "", image_started_at: float = None):
        data = {"progress": done}
        if total is not None:
            data["total"] = total
        if current_image:
            data["current_image"] = current_image
        if image_started_at is not None:
            data["image_started_at"] = image_started_at
        with open(self.progress_file, "w") as f:
            json.dump(data, f)

    def load_progress(self):
        if not self.progress_file.exists():
            return {"progress": 0}
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
