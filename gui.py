import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Combobox, Style
import threading
import os
from detect import run  # Make sure 'run' from detect.py is imported


class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Badminton Serve Fault Detection")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        # Set overall background color
        self.root.configure(bg="#f0f8ff")  # Light blue background

        # Create a style for ttk widgets
        self.style = Style()
        self.style.theme_use("clam")

        # Define custom styles
        self.style.configure("TLabel", font=("Helvetica", 11), background="#f0f8ff")
        self.style.configure("TButton", font=("Helvetica", 10, "bold"), background="#4682b4", foreground="white")
        self.style.map("TButton", background=[("active", "#5a9bd3")])
        self.style.configure("TCombobox", font=("Helvetica", 10))
        self.style.configure("TEntry", font=("Helvetica", 10))

        # Title Label
        self.title_label = tk.Label(
            root,
            text="Badminton Serve Fault Detection",
            font=("Helvetica", 16, "bold"),
            bg="#4682b4",
            fg="white",
            pady=10,
        )
        self.title_label.pack(fill="x")

        # Main Frame for inputs
        self.frame = tk.Frame(root, bg="#e6f7ff", bd=2, relief="groove")
        self.frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Height Threshold Input
        self.label_threshold = tk.Label(self.frame, text="Height Threshold (px):", bg="#e6f7ff", font=("Helvetica", 11))
        self.label_threshold.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.entry_threshold = tk.Entry(self.frame, font=("Helvetica", 10))
        self.entry_threshold.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.entry_threshold.insert(0, "540")

        # Camera Selection
        self.label_camera = tk.Label(self.frame, text="Select Camera:", bg="#e6f7ff", font=("Helvetica", 11))
        self.label_camera.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.combo_camera = Combobox(self.frame, values=self.get_camera_list(), state="readonly")
        self.combo_camera.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self.combo_camera.set("0")

        # Confidence Threshold Input
        self.label_confidence = tk.Label(self.frame, text="Confidence Threshold (%):", bg="#e6f7ff", font=("Helvetica", 11))
        self.label_confidence.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.entry_confidence = tk.Entry(self.frame, font=("Helvetica", 10))
        self.entry_confidence.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        self.entry_confidence.insert(0, "30")

        # Run Detection Button
        self.button_run = tk.Button(
            self.frame,
            text="Run Detection",
            command=self.run_detection,
            bg="#32cd32",
            fg="white",
            font=("Helvetica", 12, "bold"),
            activebackground="#228b22",
        )
        self.button_run.grid(row=3, column=0, columnspan=3, pady=20)

        # Status Label
        self.status_label = tk.Label(root, text="", bg="#f0f8ff", fg="green", font=("Helvetica", 11, "italic"))
        self.status_label.pack(pady=10)

    def get_camera_list(self):
        return [str(i) for i in range(10)]

    def run_detection(self):
        threshold = self.entry_threshold.get()
        confidence = self.entry_confidence.get()
        camera_index = self.combo_camera.get()

        # Automatically create output directory if not specified
        output_dir = os.path.join(os.getcwd(), "detection_outputs")
        os.makedirs(output_dir, exist_ok=True)

        if not threshold.isdigit():
            messagebox.showerror("Error", "Invalid height threshold!")
            return
        if not confidence.isdigit() or not (0 <= int(confidence) <= 100):
            messagebox.showerror("Error", "Confidence threshold must be between 0 and 100!")
            return

        self.status_label.config(text="Running detection...", fg="blue")
        threading.Thread(
            target=self.run_detection_thread,
            args=(int(camera_index), int(threshold), float(confidence) / 100, output_dir),
        ).start()

    def run_detection_thread(self, camera_index, threshold, confidence, output_dir):
        try:
            run(
                source=camera_index,
                project=output_dir,
                height_threshold=threshold,
                conf_thres=confidence,
                view_img=True,
            )
            self.status_label.config(text=f"Detection completed!", fg="green")
        except Exception as e:
            self.status_label.config(text="Error occurred during detection.", fg="red")
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()