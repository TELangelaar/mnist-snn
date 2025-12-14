import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from main import SimpleNN, load_data


class StepPredictor:
    def __init__(self, root, nn, images_test, labels_test):
        self.root = root
        self.root.title("MNIST Step Predictor")
        
        # Use provided model and data
        self.nn = nn
        self.images_test = images_test
        self.labels_test = labels_test
        
        # State
        # History stores the exact prediction result for a shown sample.
        # Each entry: {"data_index": int, "prediction": int, "label": int}
        self.history = []
        self.current_history_index = -1
        
        # Setup UI
        self.setup_ui()
        
        # Show first prediction (random)
        self._append_random_to_history()
        self._show_current_history_item()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Figure for matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, pady=10)
        
        # Info labels
        self.info_frame = ttk.Frame(main_frame)
        self.info_frame.grid(row=1, column=0, columnspan=3, pady=10)

        # Use a fixed-width font so spaces and digits take consistent width.
        # This prevents perceived "jumping" even when values change digits.
        self.info_font = tkfont.nametofont("TkFixedFont").copy()
        self.info_font.configure(size=12)
        self.info_font_bold = tkfont.nametofont("TkFixedFont").copy()
        self.info_font_bold.configure(size=14, weight="bold")

        # Use fixed-width labels to avoid UI "jumping" when digit counts change.
        self.index_label = ttk.Label(
            self.info_frame,
            text="",
            font=self.info_font,
            width=44,
            anchor="center",
            padding=(6, 2),
        )
        self.index_label.pack()

        self.prediction_label = ttk.Label(
            self.info_frame,
            text="",
            font=self.info_font_bold,
            width=18,
            anchor="center",
            padding=(6, 2),
        )
        self.prediction_label.pack()

        self.label_label = ttk.Label(
            self.info_frame,
            text="",
            font=self.info_font,
            width=18,
            anchor="center",
            padding=(6, 2),
        )
        self.label_label.pack()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.back_button = ttk.Button(button_frame, text="← Back", command=self.go_back, state=tk.DISABLED)
        self.back_button.pack(side=tk.LEFT, padx=5)
        
        self.forward_button = ttk.Button(button_frame, text="Forward →", command=self.go_forward)
        self.forward_button.pack(side=tk.LEFT, padx=5)
        
        self.random_button = ttk.Button(button_frame, text="Random", command=self.show_random)
        self.random_button.pack(side=tk.LEFT, padx=5)
    
    def _predict_for_index(self, data_index: int) -> tuple[int, int]:
        current_image = self.images_test[:, data_index, None]
        prediction = self.nn.make_predictions(current_image)
        predicted_digit = int(prediction[0])
        label = int(np.argmax(self.labels_test[:, data_index]))
        return predicted_digit, label

    def _append_random_to_history(self) -> None:
        data_index = int(np.random.randint(0, self.images_test.shape[1]))
        predicted_digit, label = self._predict_for_index(data_index)
        self.history.append(
            {"data_index": data_index, "prediction": predicted_digit, "label": label}
        )
        self.current_history_index = len(self.history) - 1

    def _update_button_states(self) -> None:
        self.back_button.config(
            state=tk.NORMAL if self.current_history_index > 0 else tk.DISABLED
        )

    def _show_current_history_item(self) -> None:
        if not self.history or self.current_history_index < 0:
            return

        item = self.history[self.current_history_index]
        data_index = item["data_index"]
        prediction = item["prediction"]
        label = item["label"]

        current_image = self.images_test[:, data_index, None]
        
        # Display image
        self.ax.clear()
        image_display = current_image.reshape((28, 28)) * 255
        self.ax.imshow(image_display, cmap='gray', interpolation='nearest')
        self.ax.axis('off')
        self.canvas.draw()
        
        # Update labels
        history_pos = self.current_history_index + 1
        history_len = len(self.history)
        self.index_label.config(
            text=(
                f"Index: {data_index:4d}   "
                f"History: {history_pos:2d}/{history_len:<3d}"
            )
        )
        self.prediction_label.config(
            text=f"Prediction: {prediction:2d}",
            foreground="green" if prediction == label else "red"
        )
        self.label_label.config(text=f"Actual Label: {label:2d}")

        self._update_button_states()
    
    def go_forward(self):
        if self.current_history_index < len(self.history) - 1:
            self.current_history_index += 1
            self._show_current_history_item()
            return

        self._append_random_to_history()
        self._show_current_history_item()
    
    def go_back(self):
        if self.current_history_index > 0:
            self.current_history_index -= 1
            self._show_current_history_item()
    
    def show_random(self):
        # Start a fresh timeline from a new random sample.
        self.history.clear()
        self.current_history_index = -1
        self._append_random_to_history()
        self._show_current_history_item()


def run_step_predictor(nn, images_test, labels_test):
    """Run the step predictor with provided neural network and test data."""
    root = tk.Tk()
    StepPredictor(root, nn, images_test, labels_test)
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
    root.mainloop()


def main():
    """Main function for standalone execution."""
    nn = SimpleNN()
    nn.load('./simple-nn.npz')
    _, _, images_test, labels_test = load_data()
    run_step_predictor(nn, images_test, labels_test)


if __name__ == "__main__":
    main()