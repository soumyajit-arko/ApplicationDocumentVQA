# Desktop Visual Question Answering (VQA)

A fully local, desktop-based Visual Question Answering (VQA) application using Python and PyQt6. This application allows you to upload document images (like invoices) and answers your natural language questions about the document using the local **Microsoft Phi-3.5-Vision Instruct** multimodal model.

The application runs entirely on your machine. **No data is sent to the cloud, and no API keys are required.**

---

## Prerequisites & Installation

Since the model relies natively on Python libraries (Hugging Face Transformers and PyTorch), there are **no system-level dependencies to install.**

### Setting Up the Python Environment

1. **Navigate to the project folder:**

   ```bash
   cd <path-to-your-project-folder>/docvqa
   ```
2. **(Optional but recommended) Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *This will install `PyQt6` (for the UI), `torch`, `torchvision`, `transformers`, `accelerate`, and `Pillow` (for image processing).*

---

## Running the Application

Once your Python environment is set up and dependencies are installed, you can launch the desktop application:

```bash
python main.py
```

### Usage Instructions:

1. When the app opens, click the **"Upload Document Image"** button.
2. Select an image (e.g., a PNG or JPG invoice). A responsive preview of the image will be displayed.
3. In the text box at the bottom, type your question (e.g., *"What is the total amount due?"* or *"Who is the invoice billed to?"*).
4. Click **"Ask"** or press Enter.
5. **Note on the First Run:** The application will automatically download the Phi-3.5-Vision model weights and cache them locally within a `models/` directory in the project workspace. *This may take several minutes depending on your internet connection.* Subsequent requests will load much faster since the model uses the cached files!
6. Once the model finishes thinking, the answer will seamlessly appear in the output area.

---

## Troubleshooting

- **First query is taking too long:** Ensure your internet connection is stable. The Phi-3.5-Vision model is large, and the initial download of model weights happens transparently in the background upon the first query. Look at your terminal for download progress if you launched the app via the command line.
- **Dependency Warning Popup:** If the app mentions missing dependencies, attempt to re-run `pip install -r requirements.txt` to verify all Hugging Face components and PyTorch are installed correctly.
- **Out of Memory (OOM) / Performance Issues:** The application uses `bfloat16` to reduce RAM footprint on Macs, but the model still requires significant memory and compute resources. Try closing other heavy applications running on your machine if you experience lag.
- **Where are the model weights stored?** The model weights are configured to persist directly inside the directory `models/` within the same folder as the application (`vqa_engine.py`). If you ever need to reclaim disk space, you can safely delete the `models/` directory.
