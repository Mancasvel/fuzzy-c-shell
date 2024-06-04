import tkinter as tk
from tkinter import filedialog
from c_shell import FuzzyCShell, ClusterVisualizer, read_csv_data

#visual interface implemented separated from the algorithm
class ClusterGeneratorUI:
    def __init__(self):
        self.points = None
        self.app = tk.Tk()
        self.app.title("Cluster Generator")

        tk.Label(self.app, text="Max Clusters:").grid(row=0, column=0)
        self.max_clusters_entry = tk.Entry(self.app)
        self.max_clusters_entry.grid(row=0, column=1)

        load_button = tk.Button(self.app, text="Load Data", command=self.open_file)
        load_button.grid(row=1, column=0, columnspan=2)

        generate_button = tk.Button(self.app, text="Generate Clusters", command=self.generate_clusters)
        generate_button.grid(row=2, column=0, columnspan=2)

        self.file_label = tk.Label(self.app, text="No file loaded")
        self.file_label.grid(row=3, column=0, columnspan=2)

    def open_file(self):
        file_path = filedialog.askopenfilename()
        self.points = read_csv_data(file_path)
        self.file_label.config(text=f"Loaded file: {file_path}")

    def generate_clusters(self):
        max_clusters = int(self.max_clusters_entry.get())
        model = FuzzyCShell(max_clusters=max_clusters, max_iter=100, m=2, eps=0.01, distance_metric='euclidean')
        model.fit(self.points)

        visualizer = ClusterVisualizer(model, self.points)
        visualizer.plot_clusters()

    def run(self):
        self.app.mainloop()

ui = ClusterGeneratorUI()
ui.run()
