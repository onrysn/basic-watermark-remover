import os
import subprocess
from multiprocessing import Pool

def run_inpaint(index):
    image_path = f"test_images/test ({index}).jpg"
    output_path = f"cleaned_images/cleaned ({index}).png"
    PYTHON_EXE = r"C:\Users\onrys\OneDrive\Masaüstü\Scrapper\watermark-removal\watermarkenv\Scripts\python.exe"
    cmd = [
        PYTHON_EXE, "main.py",
        "--image", image_path,
        "--output", output_path,
        "--watermark_type", "fmgproducts",
        "--checkpoint_dir", "model"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return f"Done test ({index})"

if __name__ == "__main__":
    indices = list(range(1, 41))  # test (1) to test (40)
    with Pool(processes=4) as pool:  # 4 paralel işlem
        for status in pool.imap_unordered(run_inpaint, indices):
            print(status)
