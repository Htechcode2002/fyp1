import sys
import os
import site

# Fix for ONNX Runtime GPU dependencies (CUDA/cuDNN)
def setup_cuda_dlls():
    if sys.platform == 'win32':
        # Check venv site-packages
        try:
            # Add current venv paths to environment PATH as well
            # This is often more reliable for ONNX Runtime than add_dll_directory alone
            paths_to_add = []
            
            site_packages = site.getsitepackages()
            for sp in site_packages:
                # Add NVIDIA libraries (cuDNN, CUDA Runtime, cuBLAS)
                nvidia_path = os.path.join(sp, "nvidia")
                if os.path.exists(nvidia_path):
                    for root, dirs, files in os.walk(nvidia_path):
                        if "bin" in dirs:
                            bin_path = os.path.join(root, "bin")
                            if any(f.endswith('.dll') for f in os.listdir(bin_path)):
                                os.add_dll_directory(bin_path)
                                paths_to_add.append(bin_path)
                
                # Add PyTorch libs (for zlibwapi.dll)
                torch_lib = os.path.join(sp, "torch", "lib")
                if os.path.exists(torch_lib):
                    os.add_dll_directory(torch_lib)
                    paths_to_add.append(torch_lib)
            
            # Update PATH environment variable
            if paths_to_add:
                os.environ['PATH'] = ";".join(paths_to_add) + ";" + os.environ.get('PATH', '')
        except Exception as e:
            print(f"Warning: Failed to setup CUDA DLLs: {e}")

setup_cuda_dlls()

from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.core.database import DatabaseManager

def main():
    # Ensure DB tables exist
    db = DatabaseManager()
    db.create_tables()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
