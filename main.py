import sys
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
