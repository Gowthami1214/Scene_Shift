# SceneShift: How to Run

This guide explains how to start the SceneShift application locally on Windows.

## 1. Open the Project Folder

Open PowerShell in:

```powershell
C:\Users\Gowthami\Computer Vision project
```

Or run:

```powershell
cd "C:\Users\Gowthami\Computer Vision project"
```

## 2. Install Dependencies

If dependencies are not installed yet:

```powershell
pip install -r requirements.txt
```

If you have an NVIDIA GPU and CUDA support, install the CUDA PyTorch build:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If your system has no NVIDIA GPU, the app will still run on CPU, but generation will be slower.

## 3. Start the App

Run:

```powershell
python app.py
```

When it starts successfully, open:

- Gradio UI: http://localhost:7860
- FastAPI backend: http://localhost:8000
- API docs: http://localhost:8000/docs

## 4. Use the App

1. Upload an image.
2. Choose segmentation mode. Use `Automatic (YOLO)` for normal use.
3. Enter an object prompt.
4. Enter a background prompt.
5. Choose blend mode.
6. Click `Transform Scene`.
7. Download the result using `Download Result (PNG)` or the `Result PNG` file output.

Recommended settings for your current CPU system:

- Blend Mode: `Alpha Blending` or `Poisson Blending`
- Inpainting Strength: `0.60` to `0.70`
- Guidance Scale: `7.5`
- Inference Steps: `30` in UI, internally capped to `15` on CPU
- Enable Shadow: On
- Enable Color Harmonization: On
- Use Stable Diffusion Background: On for better quality, Off for faster testing

## 5. Expected Runtime

On your current CPU setup:

- Stable Diffusion background ON: about `7-9 minutes`
- Stable Diffusion background OFF: about `3-4 minutes`

On an NVIDIA CUDA GPU:

- Expected runtime: about `30-60 seconds`

## 6. If the Port Is Already in Use

If you see an error like:

```text
Only one usage of each socket address is normally permitted
```

Find the process using the ports:

```powershell
netstat -ano | findstr ":7860"
netstat -ano | findstr ":8000"
```

Then stop the process by replacing `<PID>` with the process ID shown:

```powershell
Stop-Process -Id <PID> -Force
```

Start the app again:

```powershell
python app.py
```

## 7. Check Whether GPU Is Being Used

When the app starts, it prints the device:

```text
Device : CPU
```

or:

```text
Device : NVIDIA GPU name
CUDA   : CUDA version
```

If it says `CPU`, generation will be much slower.

## 8. Stop the App

If running in the PowerShell window, press:

```text
Ctrl + C
```

If running in the background, find and stop the Python process:

```powershell
Get-Process python
Stop-Process -Id <PID> -Force
```

