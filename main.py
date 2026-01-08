import sys
import os

# CRITICAL STABILITY FIXES: Mandatory for modern AI apps on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Prevent crashes from duplicate MKL libraries
os.environ["OMP_NUM_THREADS"] = "1"        # Prevent threading overloads in nested dependencies
os.environ["QT_API"] = "pyside6"

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

import traceback

def exception_hook(exctype, value, tb):
    """Global exception handler to catch and log all unhandled exceptions."""
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    print("\n" + "="*60)
    print("❌ UNHANDLED EXCEPTION:")
    print("="*60)
    print(error_msg)
    print("="*60 + "\n")
    
    # Also write to file for debugging
    try:
        with open("crash_log.txt", "a", encoding="utf-8") as f:
            import datetime
            f.write(f"\n{'='*60}\n")
            f.write(f"Crash at: {datetime.datetime.now()}\n")
            f.write(error_msg)
            f.write(f"{'='*60}\n")
    except:
        pass
    
    # Call the default handler
    sys.__excepthook__(exctype, value, tb)

# Install the exception hook
sys.excepthook = exception_hook

def main():
    # Ensure DB tables exist
    db = DatabaseManager()
    db.create_tables()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    # Run with exception handling
    try:
        sys.setrecursionlimit(2000) # Increase recursion limit for safety
        sys.exit(app.exec())
    except Exception as e:
        print(f"❌ Application crashed: {e}")
        traceback.print_exc()
        
        # Ensure log file is flushed
        with open("crash_log.txt", "a") as f:
            f.write(f"\nCRITICAL: Application exited due to: {e}\n")
            traceback.print_exc(file=f)
            f.flush()
        raise

if __name__ == "__main__":
    main()
