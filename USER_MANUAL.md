# 🚀 IA-Vision System - User Manual (用户启动手册)

Welcome to the **IA-Vision System**! This guide is designed for beginners to help you set up and run the AI vision system from scratch on Windows.
欢迎使用 **IA-Vision 系统**！本手册专为小白设计，教你从零开始通过命令行配置并运行这个 AI 视觉系统。

---

## 📋 Table of Contents (目录)
1. **[Step 1: Install Python (安装 Python)](#step-1-install-python-安装-python)**
2. **[Step 2: Install Git (安装 Git)](#step-2-install-git-安装-git)**
3. **[Step 3: Download the System (下载系统)](#step-3-download-the-system-下载系统)**
4. **[Step 4: Create Virtual Environment (创建虚拟环境)](#step-4-create-virtual-environment-创建虚拟环境)**
5. **[Step 5: Install Requirements (安装依赖库)](#step-5-install-requirements-安装依赖库)**
6. **[Step 6: Database Configuration (数据库配置)](#step-6-database-configuration-数据库配置)**
7. **[Step 7: Run the System (启动系统)](#step-7-run-the-system-启动系统)**
8. **[FAQ (常见问题)](#faq-常见问题)**

---

## 🛠 Step 1: Install Python (安装 Python)

The system requires **Python 3.10**.
系统需要 **Python 3.10**。

1.  **Download**: Go to [Python.org Downloads](https://www.python.org/downloads/windows/) and download the **Windows installer (64-bit)**.
    **下载**：访问 [Python 官网](https://www.python.org/downloads/windows/)，下载 **64位安装程序**。
2.  **Installation**: Run the `.exe` file.
    **安装**：运行下载好的 `.exe` 文件。
3.  **CRITICAL**: Make sure to check the box **"Add Python 3.10 to PATH"** before clicking "Install Now". This allows you to run Python from the command line.
    **至关重要**：在点击 "Install Now" 之前，务必勾选 **"Add Python 3.10 to PATH"**。这能让你在命令行中使用 Python。
4.  **Verify**: Open Command Prompt (CMD) and type `python --version`. You should see `Python 3.10.x`.
    **验证**：打开命令提示符 (CMD)，输入 `python --version`。你应该能看到 `Python 3.10.x`。

---

## 🛠 Step 2: Install Git (安装 Git)

Git is used to download the code and keep it updated.
Git 用于下载代码并进行更新。

1.  **Download**: Go to [git-scm.com](https://git-scm.com/download/win).
    **下载**：访问 [Git 官网](https://git-scm.com/download/win)。
2.  **Installation**: Follow the installer prompts and click "Next" (Default settings are fine).
    **安装**：按照安装程序提示点击“下一步”（保持默认配置即可）。
3.  **Verify**: Open CMD and type `git --version`.
    **验证**：打开 CMD 窗口，输入 `git --version`。

---

## 📂 Step 3: Download the System (下载系统)

1.  **Open CMD**: Press `Win + R`, type `cmd`, and press Enter.
    **打开命令行**：按 `Win + R` 键，输入 `cmd`，然后按回车。
2.  **Navigate**: Use `cd` command to go to your desired folder (e.g., Desktop).
    **切换目录**：使用 `cd` 命令进入你想存放项目的文件夹（例如桌面）。
    ```cmd
    cd Desktop
    ```
3.  **Clone the Repository**: Run the following command:
    **克隆项目**：输入以下命令：
    ```cmd
    git clone https://github.com/[Your-Username]/fyp1.git
    ```
4.  **Enter Folder**:
    **进入文件夹**：
    ```cmd
    cd fyp1
    ```

---

## 🏗 Step 4: Create Virtual Environment (创建虚拟环境)

A virtual environment keeps the system dependencies separate from your main computer settings.
虚拟环境可以将系统的依赖库与你电脑的主设置隔离开，防止出错。

1.  **Create venv**:
    **创建环境**：
    ```cmd
    python -m venv venv
    ```
2.  **Activate venv**:
    **激活环境**：
    ```cmd
    venv\Scripts\activate
    ```
    *(You should see `(venv)` appear at the start of your command line)*
    *(你会看到命令行开头出现了 `(venv)` 字样)*

---

## 📦 Step 5: Install Requirements (安装依赖库)

Now we install all the AI libraries needed (OpenCV, PyTorch, etc.).
现在安装所有需要的 AI 库（如 OpenCV, PyTorch 等）。

1.  **Run Install**:
    **安装命令**：
    ```cmd
    pip install -r requirements.txt
    ```
    *Note: This might take 5-10 minutes. Please do not close the window.*
    *注意：这可能需要 5-10 分钟，在此期间请勿关闭窗口。*

---

## 🏗 Step 6: Database Configuration (数据库配置)

The system uses a MySQL-compatible database (like TiDB) to store AI detection results.
系统使用 MySQL 兼容的数据库（如 TiDB）来存储 AI 检测结果。

1.  **Locate Config File**: In the `fyp1` folder, find the file named `config.json`.
    **找到配置文件**：在 `fyp1` 文件夹中，找到名为 `config.json` 的文件。
2.  **Edit Details**: Open it with Notepad and update the `db` section if you have your own database:
    **修改配置**：用记事本打开它，如果你有自己的数据库，请更新 `db` 部分：
    ```json
    "db": {
        "host": "your-database-host",
        "port": 4000,
        "user": "your-username",
        "password": "your-password",
        "database": "test"
    }
    ```
3.  **Automatic Setup**: You don't need to manually create tables. The system will automatically create the `crossing_events` table the first time you run it.
    **自动建表**：你**不需要**手动创建表。系统在第一次启动时会自动创建所需的 `crossing_events` 数据表。

4.  **Manual Import (Optional)**: If you prefer to set up the database structure manually, I have provided a `schema.sql` file in the project folder. You can import this into your database tool (like HeidiSQL, MySQL Workbench, or Navicat).
    **手动导入（可选）**：如果你想手动建立数据库结构，我在项目目录中准备了一个 `schema.sql` 文件。你可以用数据库管理工具（如 Navicat 或 MySQL Workbench）直接运行这个文件。

---

## 🚀 Step 7: Run the System (启动系统)

Every time you want to run the system, follow these 3 commands:
每次运行系统时，只需按顺序执行这 3 条命令：

1.  **Open CMD and enter project folder**:
    **进入项目文件夹**：
    ```cmd
    cd Desktop\fyp1
    ```
2.  **Activate environment**:
    **激活环境**：
    ```cmd
    venv\Scripts\activate
    ```
3.  **Launch**:
    **启动程序**：
    ```cmd
    python main.py
    ```

---

## ❓ FAQ (常见问题)

*   **Error: "python is not recognized"**
    *   **Reason**: You didn't check "Add Python to PATH".
    *   **Fix**: Re-install Python and check the box.
*   **How to stop the system?**
    *   Press `Ctrl + C` in the CMD window or simply close the CMD window.
    *   **如何停止运行？**：在命令行窗口按 `Ctrl + C` 或直接关闭窗口。
*   **Requirements installation failed?**
    *   Ensure you have a stable network. If you see "Timed out", try running the command again.

---

*Made with ❤️ for the IA-Vision Project.*
